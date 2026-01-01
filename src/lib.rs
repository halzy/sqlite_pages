//! `SQLite` page-level I/O library
//!
//! This library provides direct access to `SQLite`'s raw database pages. It operates
//! below `SQLite`'s query layer and is **not** for normal database operations - use
//! standard `SQLite` queries for reading and writing data.
//!
//! # Build Configuration
//!
//! Add the following to your `.cargo/config.toml`:
//!
//! ```toml
//! [env]
//! LIBSQLITE3_FLAGS = "-DSQLITE_ENABLE_DBPAGE_VTAB"
//! ```
//!
//! Without this flag, operations will fail with "no such table: `sqlite_dbpage`".
//!
//! # Page Numbering
//!
//! All page numbers in this API are **1-based** to match `SQLite`'s internal format.
//! The first page is page 1, not page 0.
//!
//! # ⚠️ Safety and Database Corruption Risk
//!
//! **This API bypasses `SQLite`'s safety mechanisms.** Improper use will corrupt your database.
//!
//! ## What Constitutes Valid Page Data
//!
//! Valid page data must:
//! - Come from another `SQLite` database with the same page size
//! - Maintain internal consistency (page checksums, freelist pointers, etc.)
//! - Match the target database's page size (typically 4096 bytes)
//! - Preserve the database file format (header page must be valid)
//!
//! ## What Will Cause Corruption
//!
//! - Writing arbitrary/random data to pages
//! - Modifying individual bytes within a page
//! - Mixing pages from databases with different page sizes
//! - Writing pages out of order without maintaining internal consistency
//! - Modifying the database header page (page 1) incorrectly
//!
//! ## Safe Use Cases
//!
//! This library is designed for:
//! - **Database cloning**: Copying all pages from one database to another
//! - **Database repair**: Copying valid pages from a backup
//! - **Low-level analysis**: Reading pages to understand database internals
//!
//! ## Concurrent Access
//!
//! - Multiple readers are safe (standard `SQLite` behavior)
//! - Only one writer at a time (enforced by `SQLite` locks)
//! - Use [`TransactionType::Immediate`] for writes to avoid `SQLITE_BUSY` errors
//! - Do **not** access the same database file from multiple processes while writing pages
//!
//! ## Backup Recommendations
//!
//! **Always backup your database before using this library for writes.**
//!
//! ```
//! use std::fs;
//!
//! # fn example() -> std::io::Result<()> {
//! # let tempdir = tempfile::tempdir()?;
//! # let db_path = tempdir.path().join("database.db");
//! # std::fs::write(&db_path, b"dummy")?;
//! # let backup_path = tempdir.path().join("database.db.backup");
//! // Create a backup before modifying pages
//! fs::copy(&db_path, &backup_path)?;
//! # Ok(())
//! # }
//! # example().unwrap();
//! ```
//!
//! # How to Use
//!
//! ## Reading Pages
//!
//! ```
//! use sqlite_pages::SqliteIo;
//!
//! # let tempdir = tempfile::tempdir()?;
//! # let db_path = tempdir.path().join("source.db");
//! # rusqlite::Connection::open(&db_path)?.execute("CREATE TABLE t (id INTEGER)", [])?;
//! let db = SqliteIo::new(&db_path)?;
//!
//! // Iterate over all pages
//! db.page_map(.., |page_num, data| {
//!     println!("Page {}: {} bytes", page_num, data.len());
//! })?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Writing Pages
//!
//! ```
//! use sqlite_pages::{SqliteIo, TransactionType};
//!
//! # let tempdir = tempfile::tempdir()?;
//! # let db_path = tempdir.path().join("target.db");
//! let db = SqliteIo::new(&db_path)?;
//! let mut tx = db.transaction(TransactionType::Immediate)?;
//!
//! # let pages: Vec<(usize, Vec<u8>)> = vec![];
//! for (page_num, data) in pages {
//!     tx.set_page_data(page_num, &data)?;
//! }
//!
//! tx.commit()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Async API
//!
//! For async contexts, use [`AsyncSqliteIo`]:
//!
//! ```
//! use sqlite_pages::{AsyncSqliteIo, TransactionType};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let tempdir = tempfile::tempdir()?;
//! # let db_path = tempdir.path().join("target.db");
//! let db = AsyncSqliteIo::new(&db_path).await?;
//! let tx = db.transaction(TransactionType::Immediate).await?;
//!
//! # let page_data = vec![0u8; 4096];
//! tx.set_page_data(1, &page_data).await?;
//! tx.commit().await?;
//! # Ok(())
//! # }
//! ```

use std::{
    ops::{Bound, RangeBounds},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use rusqlite::{config::DbConfig, named_params, params, Connection};
use snafu::{ResultExt as _, Snafu};

mod async_api;
mod blocking;

// Re-export for convenience and backward compatibility
pub use async_api::{AsyncInTransaction, AsyncNormal, AsyncSqliteIo};
pub use blocking::SqliteIo;

/// Marker trait for connection states.
#[doc(hidden)]
pub trait ConnectionState {}

/// Normal connection state - not in a transaction.
#[doc(hidden)]
pub struct Normal<C> {
    pub(crate) conn: C,
    pub(crate) path: std::path::PathBuf,
    pub(crate) started_empty: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl<C> ConnectionState for Normal<C> {}

/// In-transaction connection state.
#[doc(hidden)]
pub struct InTransaction<C> {
    pub(crate) conn: C,
    pub(crate) path: std::path::PathBuf,
    pub(crate) set_page_size: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl<C> ConnectionState for InTransaction<C> {}

/// The type of transaction to begin.
///
/// `SQLite` supports three transaction types with different locking behaviors:
///
/// - `Deferred`: No lock is acquired until the first read or write operation.
///   This is the default `SQLite` behavior but can lead to `SQLITE_BUSY` errors
///   if another connection acquires a write lock before you do.
///
/// - `Immediate`: Acquires a write lock immediately, preventing other connections
///   from writing. Other connections can still read. This is recommended for
///   transactions that will perform writes.
///
/// - `Exclusive`: Acquires an exclusive lock immediately, preventing other
///   connections from both reading and writing. Use this when you need complete
///   isolation.
///
/// # Default
///
/// The default transaction type is [`Immediate`](TransactionType::Immediate), which
/// is recommended for most write operations to avoid `SQLITE_BUSY` errors.
///
/// ```
/// use sqlite_pages::TransactionType;
///
/// let tx_type = TransactionType::default();
/// assert_eq!(tx_type, TransactionType::Immediate);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransactionType {
    /// Deferred transaction - lock acquired on first access
    Deferred,
    /// Immediate transaction - write lock acquired immediately (recommended for writes)
    ///
    /// This is the default transaction type.
    #[default]
    Immediate,
    /// Exclusive transaction - exclusive lock acquired immediately
    Exclusive,
}

impl TransactionType {
    /// Returns the SQL statement to begin this type of transaction.
    fn as_sql(self) -> &'static str {
        match self {
            TransactionType::Deferred => "BEGIN DEFERRED",
            TransactionType::Immediate => "BEGIN IMMEDIATE",
            TransactionType::Exclusive => "BEGIN EXCLUSIVE",
        }
    }
}

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to open database {}: {}", path.display(), source))]
    OpenDatabase {
        path: PathBuf,
        source: rusqlite::Error,
    },

    #[snafu(display("Transaction failed: {}", source))]
    Transaction { source: rusqlite::Error },

    #[snafu(display("SQL error: {}", source))]
    Sql { source: rusqlite::Error },

    #[snafu(display("Failed to update page {}: {}", page_number, source))]
    UpdatePage {
        page_number: usize,
        source: rusqlite::Error,
    },

    #[snafu(display("Failed to get page {}: {}", page_number, source))]
    GetPage {
        page_number: usize,
        source: rusqlite::Error,
    },

    #[snafu(display("Page {} not found", page_number))]
    PageNotFound { page_number: usize },

    #[snafu(display("Schema error: {}", source))]
    Schema { source: rusqlite::Error },

    #[snafu(display("Invalid page range: start={} end={}", start, end))]
    InvalidRange { start: usize, end: usize },

    #[snafu(display(
        "Invalid page size: {} (must be power of two between {} and {})",
        actual,
        min,
        max
    ))]
    InvalidPageSize {
        actual: usize,
        min: usize,
        max: usize,
    },

    #[snafu(display("Join error: {}", source))]
    Join { source: tokio::task::JoinError },

    #[snafu(display("Async connection mutex was poisoned"))]
    MutexPoisoned,

    #[snafu(display("Async connection inner value was already taken"))]
    InnerAlreadyTaken,
}

/// Represents a single database page.
///
/// A page contains both its page number and the raw binary data.
/// Page numbers are **1-based** (the first page is page 1, not page 0).
/// Page sizes are typically 4096 bytes for `SQLite` databases but can vary
/// depending on the database configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct Page {
    /// The page number (1-based: first page is 1)
    pub number: usize,
    /// The raw page data (typically 4096 bytes for `SQLite`)
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("number", &self.number)
            .field("data", &format!("[{} bytes]", self.data.len()))
            .finish()
    }
}

impl Page {
    /// Create a new Page
    #[must_use]
    pub fn new(number: usize, data: Vec<u8>) -> Self {
        Page { number, data }
    }

    /// Get the page number
    #[must_use]
    pub fn number(&self) -> usize {
        self.number
    }

    /// Get the page data
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the length of the page data
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the page data is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// Internal helper functions shared between blocking and async modules

pub(crate) fn set_page_data(
    conn: &Connection,
    page_number: usize,
    data: &[u8],
) -> Result<(), Error> {
    let mut stmt = conn
        .prepare_cached("INSERT INTO sqlite_dbpage (pgno, data, schema) values (?,?,'target');")
        .with_context(|_| SqlSnafu)?;

    let changed = stmt
        .execute(params![page_number, data])
        .with_context(|_| UpdatePageSnafu { page_number })?;

    assert_eq!(changed, 1);

    Ok(())
}

pub(crate) fn get_page_data(conn: &Connection, page_number: usize) -> Result<Page, Error> {
    let mut stmt = conn
        .prepare_cached("SELECT data FROM sqlite_dbpage('target') WHERE pgno = ?;")
        .with_context(|_| SqlSnafu)?;

    let data = stmt
        .query_row([page_number], |row| row.get(0))
        .with_context(|_| GetPageSnafu { page_number })?;

    Ok(Page::new(page_number, data))
}

pub(crate) fn max_dbpage(conn: &Connection) -> Result<usize, Error> {
    let mut max_page = 0;
    conn.pragma_query(Some("target"), "page_count", |row| {
        max_page = row.get(0)?;
        Ok(())
    })
    .with_context(|_| SqlSnafu)?;
    Ok(max_page)
}

/// Configures the database connection with required settings for page-level access.
pub(crate) fn configure_db_connection(
    conn: &Connection,
    path: &std::path::Path,
) -> Result<(), Error> {
    let configs = [
        (DbConfig::SQLITE_DBCONFIG_WRITABLE_SCHEMA, "writable schema"),
        (
            DbConfig::SQLITE_DBCONFIG_ENABLE_ATTACH_CREATE,
            "attach create",
        ),
        (
            DbConfig::SQLITE_DBCONFIG_ENABLE_ATTACH_WRITE,
            "attach write",
        ),
    ];

    for (config, name) in configs {
        let enabled = conn
            .set_db_config(config, true)
            .with_context(|_| OpenDatabaseSnafu {
                path: path.to_path_buf(),
            })?;
        assert!(enabled, "{name} should be enabled");
    }

    Ok(())
}

/// Resolves range bounds into concrete start and end page numbers.
///
/// # Arguments
/// * `range` - The range bounds to resolve
/// * `max` - The maximum page number (used when end is unbounded)
///
/// # Returns
/// A tuple of (`start_page`, `end_page`)
pub(crate) fn resolve_page_range(range: impl RangeBounds<usize>, max: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 1,
    };

    let end = match range.end_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.saturating_sub(1),
        Bound::Unbounded => max,
    };

    (start, end)
}

/// Opens a connection and attaches a database for page-level access.
///
/// Returns the connection and whether the database started empty.
pub(crate) fn open_and_attach(path: &std::path::Path) -> Result<(Connection, bool), Error> {
    let conn = Connection::open_in_memory().with_context(|_| OpenDatabaseSnafu {
        path: path.to_path_buf(),
    })?;

    // Configure database with required settings
    configure_db_connection(&conn, path)?;

    conn.execute(
        "ATTACH :path AS 'target'",
        named_params! {":path": path.to_string_lossy().as_ref()},
    )
    .with_context(|_| OpenDatabaseSnafu {
        path: path.to_path_buf(),
    })?;

    // Make it so that we can modify the database schema with page updates
    conn.pragma_update(None, "writable_schema", "ON")
        .with_context(|_| SchemaSnafu)?;

    let starting_max_page = max_dbpage(&conn)?;

    Ok((conn, starting_max_page == 0))
}

/// Sets page data with optional page size initialization.
pub(crate) fn set_page_data_with_size(
    conn: &Connection,
    page_number: usize,
    data: &[u8],
    set_page_size: &Arc<AtomicBool>,
) -> Result<(), Error> {
    if set_page_size.load(Ordering::SeqCst) {
        let page_size = data.len();

        if !page_size.is_power_of_two() {
            return Err(Error::InvalidPageSize {
                actual: page_size,
                min: 512,
                max: 65536,
            });
        }

        if !(512..=65536).contains(&page_size) {
            return Err(Error::InvalidPageSize {
                actual: page_size,
                min: 512,
                max: 65536,
            });
        }

        conn.pragma_update(Some("target"), "page_size", page_size)
            .with_context(|_| SchemaSnafu)?;

        set_page_size.store(false, Ordering::SeqCst);
    }

    set_page_data(conn, page_number, data)
}

/// Maps a function over database pages in the specified range.
pub(crate) fn page_map_impl<R: RangeBounds<usize>, F: FnMut(usize, &[u8])>(
    conn: &Connection,
    range: R,
    mut fun: F,
) -> Result<(), Error> {
    let max_page = max_dbpage(conn)?;
    let (start_page, end_page) = resolve_page_range(range, max_page);

    // Validate the range
    if start_page > end_page {
        return Err(Error::InvalidRange {
            start: start_page,
            end: end_page,
        });
    }

    let mut stmt = conn
        .prepare_cached(
            "SELECT pgno, data FROM sqlite_dbpage('target') WHERE pgno >= ?1 AND pgno <= ?2;",
        )
        .with_context(|_| SqlSnafu)?;

    let mut rows = stmt
        .query([start_page, end_page])
        .with_context(|_| SqlSnafu)?;

    while let Some(row) = rows.next().with_context(|_| SqlSnafu)? {
        let page = row.get::<_, usize>(0).with_context(|_| SqlSnafu)?;
        let data = row.get::<_, Vec<u8>>(1).with_context(|_| SqlSnafu)?;
        fun(page, &data);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_page_range_inclusive() {
        // 1..=10 with max 100
        let (start, end) = resolve_page_range(1..=10, 100);
        assert_eq!(start, 1);
        assert_eq!(end, 10);
    }

    #[test]
    fn test_resolve_page_range_exclusive() {
        // 1..10 with max 100
        let (start, end) = resolve_page_range(1..10, 100);
        assert_eq!(start, 1);
        assert_eq!(end, 9);
    }

    #[test]
    fn test_resolve_page_range_from() {
        // 5.. with max 100
        let (start, end) = resolve_page_range(5.., 100);
        assert_eq!(start, 5);
        assert_eq!(end, 100);
    }

    #[test]
    fn test_resolve_page_range_to() {
        // ..20 with max 100
        let (start, end) = resolve_page_range(..20, 100);
        assert_eq!(start, 1);
        assert_eq!(end, 19);
    }

    #[test]
    fn test_resolve_page_range_to_inclusive() {
        // ..=20 with max 100
        let (start, end) = resolve_page_range(..=20, 100);
        assert_eq!(start, 1);
        assert_eq!(end, 20);
    }

    #[test]
    fn test_resolve_page_range_full() {
        // .. with max 100
        let (start, end) = resolve_page_range(.., 100);
        assert_eq!(start, 1);
        assert_eq!(end, 100);
    }

    #[test]
    fn test_resolve_page_range_empty_database() {
        // .. with max 0
        let (start, end) = resolve_page_range(.., 0);
        assert_eq!(start, 1);
        assert_eq!(end, 0);
    }

    #[test]
    fn test_resolve_page_range_single_page() {
        // 5..=5 with max 100
        let (start, end) = resolve_page_range(5..=5, 100);
        assert_eq!(start, 5);
        assert_eq!(end, 5);
    }

    #[test]
    fn test_resolve_page_range_excluded_start() {
        // Test Bound::Excluded for start (uses + 1)
        use std::ops::Bound;
        let range = (Bound::Excluded(5), Bound::Included(10));
        let (start, end) = resolve_page_range(range, 100);
        assert_eq!(start, 6); // 5 + 1
        assert_eq!(end, 10);
    }

    #[test]
    fn test_page_is_empty() {
        let empty_page = Page::new(1, vec![]);
        assert!(empty_page.is_empty());

        let non_empty_page = Page::new(1, vec![0u8; 10]);
        assert!(!non_empty_page.is_empty());
    }

    #[test]
    fn test_page_debug_fmt() {
        let page = Page::new(42, vec![0u8; 4096]);
        let debug_str = format!("{:?}", page);
        assert!(debug_str.contains("Page"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("4096 bytes"));
    }

    #[test]
    fn test_configure_db_connection() {
        use rusqlite::config::DbConfig;

        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("config_test.db");

        let conn = rusqlite::Connection::open_in_memory().unwrap();

        // This should succeed and actually configure the connection
        configure_db_connection(&conn, &db_path).unwrap();

        // Verify the configuration was actually applied by checking the settings
        let writable_schema = conn
            .db_config(DbConfig::SQLITE_DBCONFIG_WRITABLE_SCHEMA)
            .unwrap();
        assert!(writable_schema, "writable_schema should be enabled");

        let attach_create = conn
            .db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_ATTACH_CREATE)
            .unwrap();
        assert!(attach_create, "attach_create should be enabled");

        let attach_write = conn
            .db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_ATTACH_WRITE)
            .unwrap();
        assert!(attach_write, "attach_write should be enabled");
    }

    #[test]
    fn test_open_and_attach_empty_vs_non_empty() {
        let tempdir = tempfile::tempdir().unwrap();

        // Test with non-existent (empty) database
        let empty_path = tempdir.path().join("empty.db");
        let (_, started_empty) = open_and_attach(&empty_path).unwrap();
        assert!(started_empty, "New database should start empty");

        // Test with existing database that has content
        let existing_path = tempdir.path().join("existing.db");
        {
            let conn = rusqlite::Connection::open(&existing_path).unwrap();
            conn.execute("CREATE TABLE test (id INTEGER)", []).unwrap();
        }
        let (_, started_empty) = open_and_attach(&existing_path).unwrap();
        assert!(!started_empty, "Existing database should not be empty");
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_resolve_page_range_properties(
                start in 1usize..1000,
                end in 1usize..1000,
                max in 1usize..2000
            ) {
                if start <= end && end <= max {
                    let (resolved_start, resolved_end) = resolve_page_range(start..=end, max);

                    // Start should match
                    prop_assert_eq!(resolved_start, start);
                    // End should match
                    prop_assert_eq!(resolved_end, end);
                    // Range should be valid
                    prop_assert!(resolved_start <= resolved_end);
                }
            }

            #[test]
            fn test_resolve_unbounded_end(start in 1usize..100, max in 100usize..1000) {
                let (resolved_start, resolved_end) = resolve_page_range(start.., max);

                prop_assert_eq!(resolved_start, start);
                prop_assert_eq!(resolved_end, max);
                prop_assert!(resolved_start <= resolved_end);
            }

            #[test]
            fn test_resolve_unbounded_start(end in 1usize..1000, max in 1usize..2000) {
                if end <= max {
                    let (resolved_start, resolved_end) = resolve_page_range(..=end, max);

                    prop_assert_eq!(resolved_start, 1);
                    prop_assert_eq!(resolved_end, end);
                }
            }

            #[test]
            fn test_resolve_full_range(max in 1usize..1000) {
                let (resolved_start, resolved_end) = resolve_page_range(.., max);

                prop_assert_eq!(resolved_start, 1);
                prop_assert_eq!(resolved_end, max);
            }

            #[test]
            fn test_invalid_page_size(size in 0usize..1_000_000) {
                let is_valid = size.is_power_of_two() && (512..=65536).contains(&size);

                if !is_valid && size > 0 {
                    let result: Result<(), Error> = Err(Error::InvalidPageSize {
                        actual: size,
                        min: 512,
                        max: 65536,
                    });

                    // Verify error formatting includes the values
                    if let Err(e) = result {
                        let error_msg = format!("{}", e);
                        prop_assert!(error_msg.contains(&size.to_string()));
                        prop_assert!(error_msg.contains("512"));
                        prop_assert!(error_msg.contains("65536"));
                    }
                }
            }
        }
    }
}
