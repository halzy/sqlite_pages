//! Blocking/synchronous API for page-level SQLite access.

use std::{
    ops::RangeBounds,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};

use snafu::ResultExt as _;

use crate::{
    get_page_data, max_dbpage, open_and_attach, page_map_impl, set_page_data_with_size, Error,
    Page, TransactionSnafu, TransactionType,
};

/// A transaction for safely modifying database pages.
///
/// This struct provides methods for reading and writing individual database pages.
/// Changes are only persisted when the transaction is committed via [`commit`](SqliteIoTransaction::commit).
///
/// If the transaction is dropped without calling `commit()`, all changes are
/// automatically rolled back.
pub struct SqliteIoTransaction<'a> {
    conn: &'a rusqlite::Connection,
    set_page_size: Arc<AtomicBool>,
    committed: bool,
}

impl<'a> SqliteIoTransaction<'a> {
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
    ///
    /// # Panics
    ///
    /// Panics if the page size is not a power of two or is not between 512 and 65536 bytes.
    pub fn set_page_data(&mut self, page_number: usize, data: &[u8]) -> Result<(), Error> {
        set_page_data_with_size(self.conn, page_number, data, &self.set_page_size)
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
    pub fn get_page_data(&self, page_number: usize) -> Result<Page, Error> {
        get_page_data(self.conn, page_number)
    }

    /// Commits the transaction, persisting all changes to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be committed.
    pub fn commit(mut self) -> Result<(), Error> {
        self.conn
            .execute("COMMIT", [])
            .with_context(|_| TransactionSnafu)?;
        self.committed = true;
        Ok(())
    }

    /// Rolls back the transaction, discarding all changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the rollback fails.
    pub fn rollback(mut self) -> Result<(), Error> {
        self.conn
            .execute("ROLLBACK", [])
            .with_context(|_| TransactionSnafu)?;
        self.committed = true; // Mark as handled to prevent double rollback
        Ok(())
    }
}

impl Drop for SqliteIoTransaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort rollback
            let _ = self.conn.execute("ROLLBACK", []);
        }
    }
}

/// Main interface for page-level `SQLite` database access.
///
/// This struct manages a `SQLite` connection configured for direct page access.
/// The target database is attached as the 'target' schema, and all page
/// operations work on this attached database.
///
/// # Important Notes
///
/// - The connection is opened in-memory, and the actual database file is attached
/// - All page operations use the 'target' schema name
/// - The connection is configured with special permissions for page-level access
pub struct SqliteIo {
    pub(crate) conn: rusqlite::Connection,
    path: PathBuf,
    started_empty: Arc<AtomicBool>,
}

impl std::fmt::Debug for SqliteIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteIo")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl SqliteIo {
    /// Opens a database for page-level access.
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
    ///
    /// # Example
    ///
    /// ```
    /// use sqlite_pages::SqliteIo;
    ///
    /// // Create a test database
    /// let tempdir = tempfile::tempdir()?;
    /// let db_path = tempdir.path().join("test.db");
    /// rusqlite::Connection::open(&db_path)?.execute(
    ///     "CREATE TABLE test (id INTEGER PRIMARY KEY)",
    ///     [],
    /// )?;
    ///
    /// let db = SqliteIo::new(&db_path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self, Error> {
        let path = db_path.as_ref().to_path_buf();

        let (conn, started_empty) = open_and_attach(&path)?;

        Ok(Self {
            conn,
            path,
            started_empty: Arc::new(AtomicBool::new(started_empty)),
        })
    }

    /// Starts a new transaction for page modifications.
    ///
    /// The transaction provides methods to read and write individual pages.
    /// All changes are buffered until [`commit`](SqliteIoTransaction::commit)
    /// is called. If the transaction is dropped without calling `commit()`,
    /// all changes are rolled back.
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
    /// use sqlite_pages::{SqliteIo, TransactionType};
    ///
    /// // Create a test database
    /// let tempdir = tempfile::tempdir()?;
    /// let db_path = tempdir.path().join("test.db");
    /// rusqlite::Connection::open(&db_path)?.execute(
    ///     "CREATE TABLE test (id INTEGER PRIMARY KEY)",
    ///     [],
    /// )?;
    ///
    /// let db = SqliteIo::new(&db_path)?;
    /// let mut tx = db.transaction(TransactionType::Immediate)?;
    /// let page_data = vec![0u8; 4096];
    /// tx.set_page_data(1, &page_data)?;
    /// tx.commit()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn transaction(
        &self,
        transaction_type: TransactionType,
    ) -> Result<SqliteIoTransaction<'_>, Error> {
        self.conn
            .execute(transaction_type.as_sql(), [])
            .with_context(|_| TransactionSnafu)?;

        Ok(SqliteIoTransaction {
            conn: &self.conn,
            set_page_size: Arc::clone(&self.started_empty),
            committed: false,
        })
    }

    /// Returns the number of pages in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the page count cannot be retrieved.
    pub fn page_count(&self) -> Result<usize, Error> {
        max_dbpage(&self.conn)
    }

    /// Maps a function over database pages in the specified range.
    ///
    /// Page numbers are **1-based** (the first page is page 1, not page 0).
    ///
    /// # Arguments
    ///
    /// * `range` - A range of page numbers (1-based) to process. Supports various range types:
    ///   - `1..=10` - Inclusive range from page 1 to 10
    ///   - `5..` - From page 5 to the end of the database
    ///   - `..20` - From page 1 to page 19 (exclusive end)
    ///   - `..` - All pages in the database
    /// * `fun` - A function that receives the page number (1-based) and page data for each page
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The range is invalid (start > end)
    /// - The database cannot be queried
    /// - Page data cannot be read
    ///
    /// # Example
    ///
    /// ```
    /// use sqlite_pages::SqliteIo;
    ///
    /// # let tempdir = tempfile::tempdir()?;
    /// # let db_path = tempdir.path().join("test.db");
    /// # rusqlite::Connection::open(&db_path)?.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", [])?;
    /// let db = SqliteIo::new(&db_path)?;
    /// let mut pages = Vec::new();
    /// db.page_map(1..=10, |page_num, data| {
    ///     pages.push((page_num, data.to_vec()));
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn page_map<R: RangeBounds<usize>, F: FnMut(usize, &[u8])>(
        &self,
        range: R,
        fun: F,
    ) -> Result<(), Error> {
        page_map_impl(&self.conn, range, fun)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{get_page_data, TransactionType};

    #[test]
    fn test_transaction_create_and_commit() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("transaction.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Initialize database with test data
        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 1000, 10).unwrap();

        let mut tx = db.transaction(TransactionType::Immediate).unwrap();
        let page = tx.get_page_data(1).unwrap();
        assert_eq!(page.len(), 4096);
        tx.set_page_data(1, page.data()).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_implicit_rollback_on_drop() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("rollback.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Initialize database with notes table
        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 1000, 10).unwrap();

        // Almost destroy the data in the first page, drop the transaction before it is committed
        {
            let mut tx = db.transaction(TransactionType::Immediate).unwrap();
            let empty = vec![0u8; 4096];

            tx.set_page_data(1, &empty).unwrap();

            // Transaction dropped here without commit
            drop(tx);
        }

        // row count should be 1000 (transaction was rolled back)
        let final_count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM notes;", [], |row| row.get(0))
            .unwrap();

        assert_eq!(final_count, 1000);
    }

    #[test]
    fn test_database_persistence_across_connections() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("persist.db");
        let test_data = vec![0xAAu8; 4096];

        // Create database table
        {
            let db1 = SqliteIo::new(&db_path).unwrap();

            // If the database does not need 7 pages then it will not keep the test data when we commit
            sqlite_test_utils::init_test_db(&db1.conn, "target", 1634, 1000, 10).unwrap();

            {
                let mut tx = db1.transaction(TransactionType::Immediate).unwrap();
                tx.set_page_data(7, &test_data).unwrap();
                tx.commit().unwrap();
            }
        } // db1 is dropped here

        // Read data with second connection
        let db2 = SqliteIo::new(&db_path).unwrap();
        let retrieved_page = get_page_data(&db2.conn, 7).unwrap();

        assert_eq!(retrieved_page.data(), &test_data[..]);
    }

    #[test]
    fn test_page_map_basic() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("map_test.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Create more data to ensure we have multiple pages
        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 1000, 10).unwrap();

        // Get existing pages
        let max_page = max_dbpage(&db.conn).unwrap();

        assert!(max_page >= 2, "Should have at least 2 pages");

        // Map over all pages and count them
        let mut page_count = 0;
        db.page_map(1.., |_page_num, _data| {
            page_count += 1;
        })
        .unwrap();

        assert_eq!(page_count, max_page, "Should process all pages");
    }

    #[test]
    fn test_page_map_empty_database() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("empty_map_test.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Create a minimal table to ensure database is initialized with two pages
        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 0, 0).unwrap();

        let max_pages = max_dbpage(&db.conn).unwrap();

        // Map over all pages and collect them
        let mut collected_pages: Vec<Page> = Vec::new();
        db.page_map(1..=max_pages, |page_num, data| {
            collected_pages.push(Page::new(page_num, data.to_vec()));
        })
        .unwrap();

        // database with a table should have at least two pages
        assert_eq!(
            collected_pages.len(),
            max_pages,
            "Database with table should have two pages"
        );
        assert_eq!(
            collected_pages[0].number(),
            1,
            "First page should be page 1"
        );
        assert_eq!(collected_pages[0].data().len(), 4096);
        assert_eq!(
            collected_pages[1].number(),
            2,
            "Second page should be page 2"
        );
    }

    #[test]
    fn test_page_map_ascending_order() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("sequential_map_test.db");
        let db = SqliteIo::new(&db_path).unwrap();

        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();

        let max_page = max_dbpage(&db.conn).unwrap();

        // Collect all page numbers as we process them
        let mut page_numbers: Vec<usize> = Vec::new();
        db.page_map(1..=max_page, |page_num, _data| {
            page_numbers.push(page_num);
        })
        .unwrap();

        // page numbers should start from 1
        assert!(page_numbers.contains(&1), "Should contain page 1");

        // Check that page numbers are in sequential order
        for i in 1..page_numbers.len() {
            assert!(
                page_numbers[i] > page_numbers[i - 1],
                "Page numbers should be in ascending order"
            );
        }
    }

    #[test]
    fn test_page_map_with_transaction() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("tx_map_test.db");
        let db = SqliteIo::new(&db_path).unwrap();

        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 1000, 10).unwrap();

        // Find an existing page (not page 1 which is the header)
        let existing_pages: Vec<usize> = {
            let mut stmt = db
                .conn
                .prepare("SELECT pgno FROM sqlite_dbpage('target') WHERE pgno > 1 ORDER BY pgno")
                .unwrap();
            stmt.query_map([], |row| row.get(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };

        // Make sure we have at least one non-header page to work with
        assert!(
            !existing_pages.is_empty(),
            "Should have at least one non-header page"
        );

        let test_page_num = existing_pages[0];
        let test_data = vec![0xEEu8; 4096];

        // Modify the page in a transaction
        {
            let mut tx = db.transaction(TransactionType::Immediate).unwrap();
            tx.set_page_data(test_page_num, &test_data).unwrap();
            tx.commit().unwrap();
        }

        let max_page = max_dbpage(&db.conn).unwrap();

        // Map over pages and find the specific page
        let mut found_page: Option<Page> = None;
        db.page_map(1..=max_page, |page_num, data| {
            if page_num == test_page_num {
                found_page = Some(Page::new(page_num, data.to_vec()));
            }
        })
        .unwrap();

        assert!(found_page.is_some(), "Should find page {}", test_page_num);
        let page = found_page.unwrap();
        assert_eq!(page.number(), test_page_num, "Page number should match");
        assert_eq!(
            page.data(),
            test_data,
            "page data should match what was written"
        );
    }

    #[test]
    fn test_clone_database() {
        // Create source database and initialize it with init_test_db
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let source_db_path = temppath.join("source_init.db");

        // Create and initialize source database with test data
        let source_db = SqliteIo::new(&source_db_path).unwrap();
        sqlite_test_utils::init_test_db(&source_db.conn, "target", 1634, 1000, 10).unwrap();

        // Read and store all pages from source database
        let max_page = max_dbpage(&source_db.conn).unwrap();
        let mut source_pages = Vec::new();
        source_db
            .page_map(1..=max_page, |page_num, data| {
                source_pages.push((page_num, data.to_vec()));
            })
            .unwrap();

        assert!(
            !source_pages.is_empty(),
            "Should have successfully read pages from source"
        );

        // Create destination database
        let dest_db_path = temppath.join("dest_clone.db");
        let dest_db = SqliteIo::new(&dest_db_path).unwrap();

        // An observer process that will check to see if the new table is created
        let mut dest_process =
            sqlite_test_utils::Sqlite3Process::new(dest_db_path.as_path()).unwrap();
        // ensure that the observer process does not see the notes table
        let dest_table_list = dest_process.execute("PRAGMA table_list;").unwrap();
        assert!(!dest_table_list.contains("notes"));

        // Copy all other pages from source to destination
        let mut dtx = dest_db.transaction(TransactionType::Immediate).unwrap();
        for (page_num, page_data) in source_pages.iter() {
            dtx.set_page_data(*page_num, page_data).unwrap();
        }
        dtx.commit().unwrap();

        // Now that the pages are copied, we can check if the table exists
        let dest_table_list = dest_process.execute("PRAGMA table_list;").unwrap();
        assert!(dest_table_list.contains("notes"));

        drop(dest_db);
        let dest_db = SqliteIo::new(&dest_db_path).unwrap();

        // Assert that all pages were copied correctly, skip the first as the header page has
        // some bytes that are mutated when they are written to disk.
        let dtx = dest_db.transaction(TransactionType::Immediate).unwrap();
        for (page_num, page_data) in &source_pages[1..] {
            let dest_page = get_page_data(&dtx.conn, *page_num).unwrap();
            assert_eq!(
                &dest_page.data(),
                page_data,
                "Page {} data should be identical after copying to destination",
                page_num
            );
        }
        dtx.commit().unwrap();

        // Verify all notes in destination match source
        let mut dest_stmt = dest_db
            .conn
            .prepare("SELECT id, text FROM notes ORDER BY id;")
            .unwrap();

        let dest_notes: Vec<(i32, String)> = dest_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        let mut source_stmt = source_db
            .conn
            .prepare("SELECT id, text FROM notes ORDER BY id;")
            .unwrap();

        let source_notes_data: Vec<(i32, String)> = source_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(
            source_notes_data.len(),
            dest_notes.len(),
            "Source and destination should have same number of note records"
        );

        for (i, (src_note, dst_note)) in source_notes_data.iter().zip(dest_notes.iter()).enumerate()
        {
            assert_eq!(
                src_note.0, dst_note.0,
                "Note ID at index {} should match",
                i
            );
            assert_eq!(
                src_note.1, dst_note.1,
                "Note content at index {} should match",
                i
            );
        }
    }

    #[test]
    fn test_page_count() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("page_count.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Empty database has 0 pages
        assert_eq!(db.page_count().unwrap(), 0);

        // Create a table to get some pages
        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();

        let count = db.page_count().unwrap();
        assert!(
            count >= 2,
            "Database with table should have at least 2 pages"
        );
    }

    #[test]
    fn test_invalid_page_range() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("invalid_range.db");
        let db = SqliteIo::new(&db_path).unwrap();

        sqlite_test_utils::init_test_db(&db.conn, "target", 1634, 100, 5).unwrap();

        // Start > end should error
        let result = db.page_map(10..=5, |_, _| {});
        assert!(result.is_err());

        match result {
            Err(crate::Error::InvalidRange { start, end }) => {
                assert_eq!(start, 10);
                assert_eq!(end, 5);
            }
            _ => panic!("Expected InvalidRange error"),
        }
    }
}
