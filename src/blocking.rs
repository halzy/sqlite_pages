//! Blocking/synchronous API for page-level `SQLite` access.

use std::{
    ops::RangeBounds,
    path::Path,
    sync::{atomic::AtomicBool, Arc},
};

use snafu::ResultExt as _;

use crate::{
    get_page_data, max_dbpage, open_and_attach, page_map_impl, set_page_data_with_size,
    ConnectionState, Error, InTransaction, Normal, Page, TransactionSnafu, TransactionType,
};

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
///
/// # Typestate Pattern
///
/// This connection uses a typestate pattern. When you start a transaction with
/// [`transaction`](SqliteIo::transaction), the connection moves to the
/// `InTransaction` state. You must call `commit()` or `rollback()` to
/// return to the normal state.
pub struct SqliteIo<S: ConnectionState> {
    state: S,
}

impl std::fmt::Debug for SqliteIo<Normal<rusqlite::Connection>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteIo")
            .field("path", &self.state.path)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for SqliteIo<InTransaction<rusqlite::Connection>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteIo<InTransaction>")
            .field("path", &self.state.path)
            .finish_non_exhaustive()
    }
}

impl SqliteIo<Normal<rusqlite::Connection>> {
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
            state: Normal {
                conn,
                path,
                started_empty: Arc::new(AtomicBool::new(started_empty)),
            },
        })
    }

    /// Starts a new transaction for page modifications.
    ///
    /// The transaction provides methods to read and write individual pages.
    /// All changes are buffered until [`commit`](SqliteIo::<InTransaction>::commit)
    /// is called.
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
    /// let db = tx.commit()?;  // Returns connection back to normal mode
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn transaction(
        self,
        transaction_type: TransactionType,
    ) -> Result<SqliteIo<InTransaction<rusqlite::Connection>>, Error> {
        self.state
            .conn
            .execute(transaction_type.as_sql(), [])
            .with_context(|_| TransactionSnafu)?;

        Ok(SqliteIo {
            state: InTransaction {
                conn: self.state.conn,
                path: self.state.path,
                set_page_size: self.state.started_empty,
            },
        })
    }

    /// Returns the number of pages in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the page count cannot be retrieved.
    pub fn page_count(&self) -> Result<usize, Error> {
        max_dbpage(&self.state.conn)
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
        page_map_impl(&self.state.conn, range, fun)
    }

    /// Returns a reference to the underlying connection.
    ///
    /// This is primarily for testing purposes.
    #[doc(hidden)]
    pub fn conn(&self) -> &rusqlite::Connection {
        &self.state.conn
    }
}

impl SqliteIo<InTransaction<rusqlite::Connection>> {
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
        set_page_data_with_size(
            &self.state.conn,
            page_number,
            data,
            &self.state.set_page_size,
        )
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
        get_page_data(&self.state.conn, page_number)
    }

    /// Returns the number of pages in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the page count cannot be retrieved.
    pub fn page_count(&self) -> Result<usize, Error> {
        max_dbpage(&self.state.conn)
    }

    /// Maps a function over database pages in the specified range.
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
    pub fn page_map<R: RangeBounds<usize>, F: FnMut(usize, &[u8])>(
        &self,
        range: R,
        fun: F,
    ) -> Result<(), Error> {
        page_map_impl(&self.state.conn, range, fun)
    }

    /// Commits the transaction, persisting all changes to the database.
    ///
    /// Returns the connection back to normal mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be committed.
    pub fn commit(self) -> Result<SqliteIo<Normal<rusqlite::Connection>>, Error> {
        self.state
            .conn
            .execute("COMMIT", [])
            .with_context(|_| TransactionSnafu)?;

        Ok(SqliteIo {
            state: Normal {
                conn: self.state.conn,
                path: self.state.path,
                started_empty: self.state.set_page_size,
            },
        })
    }

    /// Rolls back the transaction, discarding all changes.
    ///
    /// Returns the connection back to normal mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the rollback fails.
    pub fn rollback(self) -> Result<SqliteIo<Normal<rusqlite::Connection>>, Error> {
        self.state
            .conn
            .execute("ROLLBACK", [])
            .with_context(|_| TransactionSnafu)?;

        Ok(SqliteIo {
            state: Normal {
                conn: self.state.conn,
                path: self.state.path,
                started_empty: self.state.set_page_size,
            },
        })
    }

    /// Returns a reference to the underlying connection.
    ///
    /// This is primarily for testing purposes.
    #[doc(hidden)]
    pub fn conn(&self) -> &rusqlite::Connection {
        &self.state.conn
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
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 1000, 10).unwrap();

        let mut tx = db.transaction(TransactionType::Immediate).unwrap();
        let page = tx.get_page_data(1).unwrap();
        assert_eq!(page.len(), 4096);
        tx.set_page_data(1, page.data()).unwrap();
        let _db = tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_rollback() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("rollback.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Initialize database with notes table
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 1000, 10).unwrap();

        // Modify the first page, then rollback
        let mut tx = db.transaction(TransactionType::Immediate).unwrap();
        let empty = vec![0u8; 4096];
        tx.set_page_data(1, &empty).unwrap();
        let db = tx.rollback().unwrap();

        // row count should be 1000 (transaction was rolled back)
        let final_count: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM target.notes;", [], |row| row.get(0))
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
            sqlite_test_utils::init_test_db(db1.conn(), "target", 1634, 1000, 10).unwrap();

            let mut tx = db1.transaction(TransactionType::Immediate).unwrap();
            tx.set_page_data(7, &test_data).unwrap();
            let _db1 = tx.commit().unwrap();
        } // db1 is dropped here

        // Read data with second connection
        let db2 = SqliteIo::new(&db_path).unwrap();
        let retrieved_page = get_page_data(db2.conn(), 7).unwrap();

        assert_eq!(retrieved_page.data(), &test_data[..]);
    }

    #[test]
    fn test_page_map_basic() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("map_test.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Create more data to ensure we have multiple pages
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 1000, 10).unwrap();

        // Get existing pages
        let max_page = max_dbpage(db.conn()).unwrap();

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
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 0, 0).unwrap();

        let max_pages = max_dbpage(db.conn()).unwrap();

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

        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        let max_page = max_dbpage(db.conn()).unwrap();

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

        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 1000, 10).unwrap();

        // Find an existing page (not page 1 which is the header)
        let existing_pages: Vec<usize> = {
            let mut stmt = db
                .conn()
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
        let mut tx = db.transaction(TransactionType::Immediate).unwrap();
        tx.set_page_data(test_page_num, &test_data).unwrap();
        let db = tx.commit().unwrap();

        let max_page = max_dbpage(db.conn()).unwrap();

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
        sqlite_test_utils::init_test_db(source_db.conn(), "target", 1634, 1000, 10).unwrap();

        // Read and store all pages from source database
        let max_page = max_dbpage(source_db.conn()).unwrap();
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
        let dest_db = dtx.commit().unwrap();

        // Now that the pages are copied, we can check if the table exists
        let dest_table_list = dest_process.execute("PRAGMA table_list;").unwrap();
        assert!(dest_table_list.contains("notes"));

        drop(dest_db);
        let dest_db = SqliteIo::new(&dest_db_path).unwrap();

        // Assert that all pages were copied correctly, skip the first as the header page has
        // some bytes that are mutated when they are written to disk.
        let dtx = dest_db.transaction(TransactionType::Immediate).unwrap();
        for (page_num, page_data) in &source_pages[1..] {
            let dest_page = get_page_data(dtx.conn(), *page_num).unwrap();
            assert_eq!(
                &dest_page.data(),
                page_data,
                "Page {} data should be identical after copying to destination",
                page_num
            );
        }
        let dest_db = dtx.commit().unwrap();

        // Verify all notes in destination match source
        let mut dest_stmt = dest_db
            .conn()
            .prepare("SELECT id, text FROM target.notes ORDER BY id;")
            .unwrap();

        let dest_notes: Vec<(i32, String)> = dest_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        let mut source_stmt = source_db
            .conn()
            .prepare("SELECT id, text FROM target.notes ORDER BY id;")
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
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        let count = db.page_count().unwrap();
        assert!(
            count >= 2,
            "Database with table should have at least 2 pages"
        );
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_invalid_page_range() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("invalid_range.db");
        let db = SqliteIo::new(&db_path).unwrap();

        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

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

    #[test]
    fn test_transaction_explicit_rollback() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("explicit_rollback.db");
        let db = SqliteIo::new(&db_path).unwrap();

        // Initialize database
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        // Use external sqlite3 process to observe database state
        let mut observer = sqlite_test_utils::Sqlite3Process::new(&db_path).unwrap();

        // Get initial count
        let initial_output = observer.execute("SELECT COUNT(*) FROM notes;").unwrap();
        let initial_count: i64 = initial_output.trim().parse().unwrap();

        // Start a transaction, insert data, then rollback
        let tx = db.transaction(TransactionType::Immediate).unwrap();

        // Insert a row directly using the connection
        tx.conn()
            .execute("INSERT INTO target.notes (text) VALUES ('test')", [])
            .unwrap();

        // Now rollback
        let _db = tx.rollback().unwrap();

        // Immediately check count using external observer - this happens before Drop
        // If rollback() returned Ok(()) without executing ROLLBACK, the insert would still be visible
        let after_output = observer.execute("SELECT COUNT(*) FROM notes;").unwrap();
        let after_count: i64 = after_output.trim().parse().unwrap();

        assert_eq!(
            after_count, initial_count,
            "Rollback should have discarded the insert"
        );
    }

    #[test]
    fn test_sqlite_io_debug_fmt() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("debug_test.db");

        let db = SqliteIo::new(&db_path).unwrap();
        let debug_str = format!("{:?}", db);
        assert!(debug_str.contains("SqliteIo"));
        assert!(debug_str.contains("debug_test.db"));
    }

    #[test]
    fn test_page_map_boundary_condition() {
        let tempdir = tempfile::tempdir().unwrap();
        let temppath = tempdir.path();
        let db_path = temppath.join("boundary_test.db");
        let db = SqliteIo::new(&db_path).unwrap();

        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        let max_page = max_dbpage(db.conn()).unwrap();

        // Test exact boundary - should include max_page
        let mut pages_found = 0;
        db.page_map(max_page..=max_page, |page_num, _| {
            assert_eq!(page_num, max_page);
            pages_found += 1;
        })
        .unwrap();
        assert_eq!(pages_found, 1, "Should find exactly one page at boundary");

        // Test range that goes beyond max - should still work
        let mut pages_in_range = 0;
        db.page_map(1..=max_page, |_, _| {
            pages_in_range += 1;
        })
        .unwrap();
        assert_eq!(pages_in_range, max_page, "Should find all pages up to max");
    }

    #[test]
    fn test_typestate_prevents_reuse() {
        // This test verifies the typestate pattern works correctly
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("typestate_test.db");

        let db = SqliteIo::new(&db_path).unwrap();
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        // Start transaction - db is consumed
        let tx = db.transaction(TransactionType::Immediate).unwrap();

        // Commit returns a new SqliteIo<Normal>
        let db = tx.commit().unwrap();

        // Can start another transaction
        let tx = db.transaction(TransactionType::Immediate).unwrap();

        // Rollback also returns a SqliteIo<Normal>
        let _db = tx.rollback().unwrap();
    }

    #[test]
    fn test_concurrent_readers() {
        use std::sync::Arc;
        use std::thread;

        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("concurrent_readers.db");

        // Set up database
        {
            let db = SqliteIo::new(&db_path).unwrap();
            sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();
        }

        let db_path = Arc::new(db_path);
        let mut handles = vec![];

        // Spawn multiple reader threads
        for i in 0..5 {
            let db_path = Arc::clone(&db_path);
            let handle = thread::spawn(move || {
                let db = SqliteIo::new(db_path.as_ref()).unwrap();
                let mut count = 0;

                // Each thread reads all pages
                db.page_map(.., |_num, _data| {
                    count += 1;
                })
                .unwrap();

                (i, count)
            });
            handles.push(handle);
        }

        // All threads should complete successfully
        for handle in handles {
            let (thread_id, count) = handle.join().unwrap();
            assert!(count >= 2, "Thread {} should have read pages", thread_id);
        }
    }

    #[test]
    fn test_concurrent_read_write_isolation() {
        use std::sync::{Arc, Barrier};
        use std::thread;
        use std::time::Duration;

        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("concurrent_rw.db");

        // Initialize database
        {
            let db = SqliteIo::new(&db_path).unwrap();
            sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();
        }

        let db_path = Arc::new(db_path);
        let barrier = Arc::new(Barrier::new(2));

        // Writer thread
        let db_path_writer = Arc::clone(&db_path);
        let barrier_writer = Arc::clone(&barrier);
        let writer = thread::spawn(move || {
            let db = SqliteIo::new(db_path_writer.as_ref()).unwrap();
            let mut tx = db.transaction(TransactionType::Immediate).unwrap();

            // Signal that we have the write lock
            barrier_writer.wait();

            // Hold the lock briefly
            thread::sleep(Duration::from_millis(50));

            let page = tx.get_page_data(1).unwrap();
            tx.set_page_data(1, page.data()).unwrap();
            tx.commit().unwrap();
        });

        // Reader thread
        let db_path_reader = Arc::clone(&db_path);
        let barrier_reader = Arc::clone(&barrier);
        let reader = thread::spawn(move || {
            // Wait for writer to acquire lock
            barrier_reader.wait();

            // Give writer time to fully acquire the lock
            thread::sleep(Duration::from_millis(10));

            // Reader can still open a connection and read
            let db = SqliteIo::new(db_path_reader.as_ref()).unwrap();
            let mut count = 0;
            db.page_map(.., |_, _| {
                count += 1;
            })
            .unwrap();
            count
        });

        // Both should complete
        writer.join().unwrap();
        let read_count = reader.join().unwrap();
        assert!(read_count >= 2);
    }

    #[test]
    fn test_transaction_page_count() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("tx_page_count.db");

        let db = SqliteIo::new(&db_path).unwrap();
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        let tx = db.transaction(TransactionType::Immediate).unwrap();
        let count = tx.page_count().unwrap();
        assert!(count >= 2, "Should have pages in transaction");

        let _db = tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_page_map() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("tx_page_map.db");

        let db = SqliteIo::new(&db_path).unwrap();
        sqlite_test_utils::init_test_db(db.conn(), "target", 1634, 100, 5).unwrap();

        let tx = db.transaction(TransactionType::Immediate).unwrap();

        let mut pages = Vec::new();
        tx.page_map(1.., |num, data| {
            pages.push((num, data.len()));
        })
        .unwrap();

        assert!(!pages.is_empty(), "Should read pages in transaction");

        let _db = tx.commit().unwrap();
    }

    #[test]
    fn test_in_transaction_debug_fmt() {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("tx_debug_test.db");

        let db = SqliteIo::new(&db_path).unwrap();
        let tx = db.transaction(TransactionType::Immediate).unwrap();
        let debug_str = format!("{:?}", tx);
        assert!(debug_str.contains("SqliteIo<InTransaction>"));
        assert!(debug_str.contains("tx_debug_test.db"));
        let _db = tx.commit().unwrap();
    }
}
