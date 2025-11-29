//! Asynchronous API for page-level `SQLite` access using tokio.

use std::{ops::RangeBounds, path::Path, sync::Mutex};

use snafu::{OptionExt as _, ResultExt as _};

use crate::{
    blocking::SqliteIo, Error, InTransaction, InnerAlreadyTakenSnafu, JoinSnafu,
    MutexPoisonedSnafu, Normal, Page, TransactionType,
};

/// Async connection state trait.
#[doc(hidden)]
pub trait AsyncConnectionState {}

/// Generic async state wrapper that provides take/put/into_inner operations.
#[doc(hidden)]
pub struct AsyncState<T> {
    inner: Mutex<Option<T>>,
}

impl<T> AsyncState<T> {
    fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(Some(value)),
        }
    }

    fn take(&self) -> Result<T, Error> {
        self.inner
            .lock()
            .ok()
            .context(MutexPoisonedSnafu)?
            .take()
            .context(InnerAlreadyTakenSnafu)
    }

    fn put(&self, value: T) -> Result<(), Error> {
        *self.inner.lock().ok().context(MutexPoisonedSnafu)? = Some(value);
        Ok(())
    }

    fn into_inner(self) -> Result<T, Error> {
        self.inner
            .into_inner()
            .ok()
            .context(MutexPoisonedSnafu)?
            .context(InnerAlreadyTakenSnafu)
    }
}

/// Normal async connection state - wraps a blocking connection.
#[doc(hidden)]
pub type AsyncNormal = AsyncState<SqliteIo<Normal<rusqlite::Connection>>>;

impl AsyncConnectionState for AsyncNormal {}

/// In-transaction async connection state - wraps a blocking transaction.
#[doc(hidden)]
pub type AsyncInTransaction = AsyncState<SqliteIo<InTransaction<rusqlite::Connection>>>;

impl AsyncConnectionState for AsyncInTransaction {}

// Static assertions to ensure async types are Send + Sync
const _: () = {
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}

    assert_send::<AsyncSqliteIo<AsyncNormal>>();
    assert_sync::<AsyncSqliteIo<AsyncNormal>>();
    assert_send::<AsyncSqliteIo<AsyncInTransaction>>();
    assert_sync::<AsyncSqliteIo<AsyncInTransaction>>();
};

/// Async interface for page-level `SQLite` database access.
///
/// This struct wraps the blocking `SqliteIo` and executes all blocking `SQLite`
/// operations via `tokio::task::spawn_blocking`.
///
/// # Important Notes
///
/// - The connection is opened in-memory, and the actual database file is attached
/// - All page operations use the 'target' schema name
/// - The connection is configured with special permissions for page-level access
///
/// # Typestate Pattern
///
/// This connection uses a typestate pattern. When you start a transaction with
/// [`transaction`](AsyncSqliteIo::transaction), the connection moves to the
/// `AsyncInTransaction` state. You must call `commit()` or `rollback()` to
/// return to the normal state.
pub struct AsyncSqliteIo<S: AsyncConnectionState> {
    state: S,
}

impl std::fmt::Debug for AsyncSqliteIo<AsyncNormal> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncSqliteIo").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for AsyncSqliteIo<AsyncInTransaction> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncSqliteIo<InTransaction>")
            .finish_non_exhaustive()
    }
}

impl AsyncSqliteIo<AsyncNormal> {
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

        let inner = tokio::task::spawn_blocking(move || SqliteIo::new(&path))
            .await
            .with_context(|_| JoinSnafu)??;

        Ok(Self {
            state: AsyncState::new(inner),
        })
    }

    /// Starts a new async transaction for page modifications.
    ///
    /// This method consumes the connection and returns it in transaction mode.
    /// You must call `commit()` or `rollback()` on the returned connection
    /// to get back to normal mode.
    ///
    /// # Arguments
    ///
    /// * `transaction_type` - The type of transaction to begin. Use `TransactionType::Immediate`
    ///   (the default) for write operations to avoid `SQLITE_BUSY` errors.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started.
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
    /// let db = tx.commit().await?;  // Returns connection back to normal mode
    /// # Ok(())
    /// # }
    /// ```
    pub async fn transaction(
        self,
        transaction_type: TransactionType,
    ) -> Result<AsyncSqliteIo<AsyncInTransaction>, Error> {
        let inner = self.state.into_inner()?;

        let tx_inner = tokio::task::spawn_blocking(move || inner.transaction(transaction_type))
            .await
            .with_context(|_| JoinSnafu)??;

        Ok(AsyncSqliteIo {
            state: AsyncState::new(tx_inner),
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
        let inner = self.state.take()?;

        let (result, inner) = tokio::task::spawn_blocking(move || {
            let result = inner.page_map(range, fun);
            (result, inner)
        })
        .await
        .with_context(|_| JoinSnafu)?;

        self.state.put(inner)?;
        result
    }

    /// Returns the number of pages in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the page count cannot be retrieved.
    pub async fn page_count(&self) -> Result<usize, Error> {
        let inner = self.state.take()?;

        let (result, inner) = tokio::task::spawn_blocking(move || {
            let result = inner.page_count();
            (result, inner)
        })
        .await
        .with_context(|_| JoinSnafu)?;

        self.state.put(inner)?;
        result
    }
}

impl AsyncSqliteIo<AsyncInTransaction> {
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
        let data = data.to_vec();
        let mut inner = self.state.take()?;

        let (result, inner) = tokio::task::spawn_blocking(move || {
            let result = inner.set_page_data(page_number, &data);
            (result, inner)
        })
        .await
        .with_context(|_| JoinSnafu)?;

        self.state.put(inner)?;
        result
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
        let inner = self.state.take()?;

        let (result, inner) = tokio::task::spawn_blocking(move || {
            let result = inner.get_page_data(page_number);
            (result, inner)
        })
        .await
        .with_context(|_| JoinSnafu)?;

        self.state.put(inner)?;
        result
    }

    /// Returns the number of pages in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the page count cannot be retrieved.
    pub async fn page_count(&self) -> Result<usize, Error> {
        let inner = self.state.take()?;

        let (result, inner) = tokio::task::spawn_blocking(move || {
            let result = inner.page_count();
            (result, inner)
        })
        .await
        .with_context(|_| JoinSnafu)?;

        self.state.put(inner)?;
        result
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
    /// Returns an error if the range is invalid or pages cannot be read.
    pub async fn page_map<R, F>(&self, range: R, fun: F) -> Result<(), Error>
    where
        R: RangeBounds<usize> + Send + 'static,
        F: FnMut(usize, &[u8]) + Send + 'static,
    {
        let inner = self.state.take()?;

        let (result, inner) = tokio::task::spawn_blocking(move || {
            let result = inner.page_map(range, fun);
            (result, inner)
        })
        .await
        .with_context(|_| JoinSnafu)?;

        self.state.put(inner)?;
        result
    }

    /// Commits the transaction, persisting all changes to the database.
    ///
    /// Returns the connection back to normal mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be committed.
    pub async fn commit(self) -> Result<AsyncSqliteIo<AsyncNormal>, Error> {
        let inner = self.state.into_inner()?;

        let normal_inner = tokio::task::spawn_blocking(move || inner.commit())
            .await
            .with_context(|_| JoinSnafu)??;

        Ok(AsyncSqliteIo {
            state: AsyncState::new(normal_inner),
        })
    }

    /// Rolls back the transaction, discarding all changes.
    ///
    /// Returns the connection back to normal mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the rollback fails.
    pub async fn rollback(self) -> Result<AsyncSqliteIo<AsyncNormal>, Error> {
        let inner = self.state.into_inner()?;

        let normal_inner = tokio::task::spawn_blocking(move || inner.rollback())
            .await
            .with_context(|_| JoinSnafu)??;

        Ok(AsyncSqliteIo {
            state: AsyncState::new(normal_inner),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransactionType;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Creates a test database and returns the path
    fn create_test_db() -> (tempfile::TempDir, PathBuf) {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("test.db");
        let db = crate::blocking::SqliteIo::new(&db_path).unwrap();
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();
        (tempdir, db_path)
    }

    #[tokio::test]
    async fn test_multiple_operations_on_normal() {
        let (_tempdir, db_path) = create_test_db();
        let db = AsyncSqliteIo::new(&db_path).await.unwrap();

        // Test page_count multiple times (verifies put() works)
        let count1 = db.page_count().await.unwrap();
        let count2 = db.page_count().await.unwrap();
        assert_eq!(count1, count2);
        assert!(count1 >= 2);

        // Test page_map
        let pages = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pages_clone = std::sync::Arc::clone(&pages);
        db.page_map(1.., move |_, _| {
            pages_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
        .await
        .unwrap();
        assert!(pages.load(std::sync::atomic::Ordering::SeqCst) > 0);
    }

    #[tokio::test]
    async fn test_transaction_commit_and_persist() {
        let (_tempdir, db_path) = create_test_db();
        let db = AsyncSqliteIo::new(&db_path).await.unwrap();

        // Write data and commit
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let test_data = vec![0xABu8; 4096];
        tx.set_page_data(2, &test_data).await.unwrap();
        let read_back = tx.get_page_data(2).await.unwrap();
        assert_eq!(read_back.data(), &test_data[..]);
        let db = tx.commit().await.unwrap();

        // Verify persistence
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let after = tx.get_page_data(2).await.unwrap();
        assert_eq!(after.data(), &test_data[..]);
        let _db = tx.commit().await.unwrap();
    }

    #[tokio::test]
    async fn test_transaction_rollback() {
        let (_tempdir, db_path) = create_test_db();
        let db = AsyncSqliteIo::new(&db_path).await.unwrap();

        // Get original data
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let original = tx.get_page_data(1).await.unwrap();
        let db = tx.commit().await.unwrap();

        // Modify and rollback
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        tx.set_page_data(1, &vec![0u8; 4096]).await.unwrap();
        let db = tx.rollback().await.unwrap();

        // Verify unchanged
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        let after = tx.get_page_data(1).await.unwrap();
        assert_eq!(original.data(), after.data());
        let _db = tx.commit().await.unwrap();
    }

    #[tokio::test]
    async fn test_transaction_types() {
        let (_tempdir, db_path) = create_test_db();

        // Test all transaction types
        for tx_type in [
            TransactionType::Deferred,
            TransactionType::Immediate,
            TransactionType::Exclusive,
        ] {
            let db = AsyncSqliteIo::new(&db_path).await.unwrap();
            let tx = db.transaction(tx_type).await.unwrap();
            let _count = tx.page_count().await.unwrap();
            let _db = tx.commit().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_transaction_page_map_and_count() {
        let (_tempdir, db_path) = create_test_db();
        let db = AsyncSqliteIo::new(&db_path).await.unwrap();
        let tx = db.transaction(TransactionType::Immediate).await.unwrap();

        // Test page_count on transaction
        let count = tx.page_count().await.unwrap();
        assert!(count >= 2);

        // Test page_map on transaction
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mapped_clone = std::sync::Arc::clone(&mapped);
        tx.page_map(1.., move |_, _| {
            mapped_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
        .await
        .unwrap();
        assert_eq!(mapped.load(std::sync::atomic::Ordering::SeqCst), count);
        let _db = tx.commit().await.unwrap();
    }

    #[tokio::test]
    async fn test_debug_fmt() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("debug.db");

        let db = AsyncSqliteIo::new(&db_path).await.unwrap();
        assert!(format!("{:?}", db).contains("AsyncSqliteIo"));

        let tx = db.transaction(TransactionType::Immediate).await.unwrap();
        assert!(format!("{:?}", tx).contains("InTransaction"));
        let _db = tx.commit().await.unwrap();
    }

    #[test]
    fn test_mutex_poisoned_error() {
        let state: AsyncNormal = AsyncState::new(
            crate::blocking::SqliteIo::new(tempfile::tempdir().unwrap().path().join("poison.db"))
                .unwrap(),
        );

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = state.inner.lock().unwrap();
            panic!("poison mutex");
        }));

        assert!(matches!(state.take().unwrap_err(), Error::MutexPoisoned));
    }

    #[test]
    fn test_inner_already_taken_error() {
        let state: AsyncNormal = AsyncState {
            inner: Mutex::new(None),
        };
        assert!(matches!(
            state.take().unwrap_err(),
            Error::InnerAlreadyTaken
        ));

        let state: AsyncInTransaction = AsyncState {
            inner: Mutex::new(None),
        };
        assert!(matches!(
            state.take().unwrap_err(),
            Error::InnerAlreadyTaken
        ));
    }
}
