//! Asynchronous API for page-level SQLite access using tokio.

use std::{
    ops::RangeBounds,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

use rusqlite::Connection;
use snafu::ResultExt as _;
use tokio::sync::Mutex;

use crate::{
    get_page_data, max_dbpage, open_and_attach, page_map_impl, set_page_data_with_size, Error,
    JoinSnafu, Page, TransactionSnafu, TransactionType,
};

/// Async interface for page-level `SQLite` database access.
///
/// This struct wraps a `SQLite` connection in an `Arc<Mutex<>>` allowing it to be
/// cloned and shared across async tasks. All blocking SQLite operations are executed
/// via `tokio::task::spawn_blocking`.
///
/// # Important Notes
///
/// - The connection is opened in-memory, and the actual database file is attached
/// - All page operations use the 'target' schema name
/// - The connection is configured with special permissions for page-level access
/// - This struct is `Clone` and can be shared across tasks
#[derive(Clone)]
pub struct AsyncSqliteIo {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
    started_empty: Arc<AtomicBool>,
}

impl std::fmt::Debug for AsyncSqliteIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncSqliteIo")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl AsyncSqliteIo {
    /// Opens a database for page-level access asynchronously.
    ///
    /// This creates an in-memory `SQLite` connection and attaches the database
    /// file at `db_path` as the 'target' schema. The connection is configured
    /// with special permissions to allow page-level manipulation.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the `SQLite` database file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The database path is invalid
    /// - The database cannot be opened or attached
    /// - Required database configurations cannot be set
    pub async fn new<P: AsRef<Path>>(db_path: P) -> Result<Self, Error> {
        let path = db_path.as_ref().to_path_buf();

        let path_clone = path.clone();
        let (conn, started_empty) =
            tokio::task::spawn_blocking(move || open_and_attach(&path_clone))
                .await
                .with_context(|_| JoinSnafu)??;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path,
            started_empty: Arc::new(AtomicBool::new(started_empty)),
        })
    }

    /// Starts a new async transaction for page modifications.
    ///
    /// The transaction holds the connection lock for its entire lifetime.
    /// All operations on the transaction are async and use `spawn_blocking` internally.
    /// You must call `commit()` to persist changes; dropping without commit will rollback.
    ///
    /// # Arguments
    ///
    /// * `transaction_type` - The type of transaction to begin. Use `TransactionType::Immediate`
    ///   (the default) for write operations to avoid `SQLITE_BUSY` errors.
    ///
    /// # Example
    ///
    /// ```
    /// use sqlite_pages::{AsyncSqliteIo, TransactionType};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a test database
    /// let tempdir = tempfile::tempdir()?;
    /// let db_path = tempdir.path().join("test.db");
    /// rusqlite::Connection::open(&db_path)?.execute(
    ///     "CREATE TABLE test (id INTEGER PRIMARY KEY)",
    ///     [],
    /// )?;
    ///
    /// let db = AsyncSqliteIo::new(&db_path).await?;
    /// let tx = db.transaction(TransactionType::Immediate).await?;
    ///
    /// // Can await between operations
    /// let page = tx.get_page_data(1).await?;
    /// tx.set_page_data(1, page.data()).await?;
    /// tx.commit().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn transaction(
        &self,
        transaction_type: TransactionType,
    ) -> Result<AsyncTransaction, Error> {
        let conn = Arc::clone(&self.conn);
        let started_empty = Arc::clone(&self.started_empty);

        // Start transaction in blocking context
        tokio::task::spawn_blocking({
            let conn_clone = Arc::clone(&conn);
            move || {
                let guard = conn_clone.blocking_lock();

                guard
                    .execute(transaction_type.as_sql(), [])
                    .with_context(|_| TransactionSnafu)?;

                Ok::<_, Error>(())
            }
        })
        .await
        .with_context(|_| JoinSnafu)??;

        Ok(AsyncTransaction {
            conn,
            set_page_size: started_empty,
            committed: false,
        })
    }

    /// Maps a function over database pages in the specified range asynchronously.
    ///
    /// Page numbers are **1-based** (the first page is page 1, not page 0).
    ///
    /// # Arguments
    ///
    /// * `range` - A range of page numbers (1-based) to process
    /// * `fun` - A function that receives the page number (1-based) and page data for each page
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The range is invalid (start > end)
    /// - The database cannot be queried
    /// - Page data cannot be read
    pub async fn page_map<R, F>(&self, range: R, fun: F) -> Result<(), Error>
    where
        R: RangeBounds<usize> + Send + 'static,
        F: FnMut(usize, &[u8]) + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let guard = conn.blocking_lock();
            page_map_impl(&*guard, range, fun)
        })
        .await
        .with_context(|_| JoinSnafu)?
    }

    /// Returns the number of pages in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the page count cannot be retrieved.
    pub async fn page_count(&self) -> Result<usize, Error> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let guard = conn.blocking_lock();
            max_dbpage(&*guard)
        })
        .await
        .with_context(|_| JoinSnafu)?
    }
}

/// An async transaction for page modifications.
///
/// This transaction holds the connection lock and provides async methods for
/// reading and writing pages. Each operation uses `spawn_blocking` internally.
/// You must call `commit()` to persist changes; dropping without commit will rollback.
pub struct AsyncTransaction {
    conn: Arc<Mutex<Connection>>,
    set_page_size: Arc<AtomicBool>,
    committed: bool,
}

impl AsyncTransaction {
    /// Sets the data for a specific page in the database.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The page number (1-based: first page is 1)
    /// * `data` - The raw page data
    ///
    /// # Errors
    ///
    /// Returns an error if the page data cannot be written to the database.
    pub async fn set_page_data(&self, page_number: usize, data: &[u8]) -> Result<(), Error> {
        let conn = Arc::clone(&self.conn);
        let data = data.to_vec();
        let set_page_size = Arc::clone(&self.set_page_size);

        tokio::task::spawn_blocking(move || {
            let guard = conn.blocking_lock();
            set_page_data_with_size(&*guard, page_number, &data, &set_page_size)
        })
        .await
        .with_context(|_| JoinSnafu)?
    }

    /// Retrieves the data for a specific page from the database.
    ///
    /// # Arguments
    ///
    /// * `page_number` - The page number (1-based: first page is 1)
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be read from the database.
    pub async fn get_page_data(&self, page_number: usize) -> Result<Page, Error> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let guard = conn.blocking_lock();
            get_page_data(&*guard, page_number)
        })
        .await
        .with_context(|_| JoinSnafu)?
    }

    /// Commits the transaction, persisting all changes to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be committed.
    pub async fn commit(mut self) -> Result<(), Error> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let guard = conn.blocking_lock();
            guard
                .execute("COMMIT", [])
                .with_context(|_| TransactionSnafu)
        })
        .await
        .with_context(|_| JoinSnafu)??;

        self.committed = true;
        Ok(())
    }

    /// Rolls back the transaction, discarding all changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the rollback fails.
    pub async fn rollback(mut self) -> Result<(), Error> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let guard = conn.blocking_lock();
            guard
                .execute("ROLLBACK", [])
                .with_context(|_| TransactionSnafu)
        })
        .await
        .with_context(|_| JoinSnafu)??;

        self.committed = true; // Mark as handled to prevent double rollback
        Ok(())
    }
}

impl Drop for AsyncTransaction {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort rollback - can't do async in drop
            let conn = Arc::clone(&self.conn);
            let _ = std::thread::spawn(move || {
                let guard = conn.blocking_lock();
                let _ = guard.execute("ROLLBACK", []);
            })
            .join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransactionType;

    #[tokio::test]
    async fn test_async_new_and_page_map() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("async_test.db");

        // Create a database first using blocking API
        {
            let db = crate::blocking::SqliteIo::new(&db_path).unwrap();
            sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();
        }

        // Now test the async API
        let db = AsyncSqliteIo::new(&db_path).await.unwrap();

        let page_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let page_count_clone = Arc::clone(&page_count);
        db.page_map(1.., move |_page_num, _data| {
            page_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
        .await
        .unwrap();

        assert!(
            page_count.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "Should have read some pages"
        );
    }

    #[tokio::test]
    async fn test_async_begin_transaction() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("async_tx_test.db");

        // Create a database first
        {
            let db = crate::blocking::SqliteIo::new(&db_path).unwrap();
            sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();
        }

        let db = AsyncSqliteIo::new(&db_path).await.unwrap();

        // Test reading and writing in a transaction
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let page = tx.get_page_data(1).await.unwrap();
        assert_eq!(page.len(), 4096);
        tx.set_page_data(1, page.data()).await.unwrap();
        tx.commit().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_transaction_rollback_on_drop() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("async_rollback_test.db");

        // Create a database first
        {
            let db = crate::blocking::SqliteIo::new(&db_path).unwrap();
            sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();
        }

        let db = AsyncSqliteIo::new(&db_path).await.unwrap();

        // Get original page data
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let original_page = tx.get_page_data(1).await.unwrap();
        tx.commit().await.unwrap();

        // Modify but don't commit
        {
            let tx = db.transaction(TransactionType::Immediate).await.unwrap();
            let empty = vec![0u8; 4096];
            tx.set_page_data(1, &empty).await.unwrap();
            // No commit - should rollback on drop
        }

        // Verify data unchanged
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let after_page = tx.get_page_data(1).await.unwrap();
        tx.commit().await.unwrap();

        assert_eq!(
            original_page.data(),
            after_page.data(),
            "Data should be unchanged after rollback"
        );
    }

    #[tokio::test]
    async fn test_async_clone_and_share() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("async_clone_test.db");

        // Create a database first
        {
            let db = crate::blocking::SqliteIo::new(&db_path).unwrap();
            sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();
        }

        let db = AsyncSqliteIo::new(&db_path).await.unwrap();
        let db_clone = db.clone();

        // Use both handles
        let max1 = db.page_count().await.unwrap();
        let max2 = db_clone.page_count().await.unwrap();

        assert_eq!(max1, max2);
    }
}
