//! Integration tests for WAL mode interactions and concurrent access patterns.
//!
//! These tests verify that sqlite_pages behaves correctly when:
//! - rusqlite connections access the same database concurrently
//! - WAL mode is enabled
//! - Multiple transactions are committed
//! - Large data spans multiple pages
//!
//! Background:
//! These tests were created to investigate database corruption issues observed
//! when replicating SQLite databases using sqlite_pages. The corruption manifested as:
//! - `PRAGMA integrity_check` returning errors
//! - Pages marked as "never used" even though data was written
//! - "database disk image is malformed" errors

use sqlite_pages::{AsyncSqliteIo, SqliteIo, TransactionType};
use std::path::Path;

/// Helper to create a WAL-mode database with rusqlite
fn create_wal_database(path: &Path) -> rusqlite::Connection {
    let conn = rusqlite::Connection::open(path).expect("Failed to open database");
    conn.pragma_update(None, "journal_mode", "wal")
        .expect("Failed to set WAL mode");
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL)",
        [],
    )
    .expect("Failed to create table");
    conn
}

/// Helper to insert test data
fn insert_test_data(conn: &rusqlite::Connection, count: usize) {
    for i in 0..count {
        conn.execute(
            "INSERT INTO notes (text) VALUES (?)",
            [format!("Note number {}", i)],
        )
        .expect("Failed to insert test data");
    }
}

/// Helper to run integrity check and return result
fn integrity_check(conn: &rusqlite::Connection) -> String {
    conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .expect("Failed to run integrity check")
}

/// Helper to verify that integrity check passes
fn assert_integrity_ok(conn: &rusqlite::Connection, context: &str) {
    let result = integrity_check(conn);
    assert_eq!(
        result, "ok",
        "Integrity check failed at {}: {}",
        context, result
    );
}

/// Test 1: Concurrent rusqlite Read During sqlite_pages Write
///
/// Verifies that writing pages with sqlite_pages while rusqlite has a read
/// connection doesn't corrupt the database.
#[tokio::test]
async fn test_concurrent_rusqlite_read_during_page_write() {
    let tempdir = tempfile::tempdir().unwrap();
    let db_path = tempdir.path().join("concurrent_read.db");

    // 1. Create a WAL-mode database with rusqlite and populate with data
    {
        let conn = create_wal_database(&db_path);
        insert_test_data(&conn, 100);
    }

    // 2. Open a read-only rusqlite connection (keep it open)
    let read_conn = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .expect("Failed to open read-only connection");

    // Verify we can read the data
    let initial_count: i64 = read_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count rows");
    assert_eq!(initial_count, 100);

    // 3. Use sqlite_pages to write new pages while rusqlite connection is open
    let source_db = SqliteIo::new(&db_path).unwrap();
    let source_page_count = source_db.page_count().unwrap();

    // Read all pages from source
    let mut pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            pages.push((page_num, data.to_vec()));
        })
        .unwrap();

    // Create a destination database and write pages while read_conn is still open
    let dest_path = tempdir.path().join("concurrent_dest.db");
    let dest_db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = dest_db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    // 4. Write all pages
    for (page_num, data) in pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }

    // 5. Commit the sqlite_pages transaction
    let _dest_db = tx.commit().await.unwrap();

    // 6. Verify rusqlite can still read from original database
    let count_after: i64 = read_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count rows after write");
    assert_eq!(count_after, 100);

    // 7. Run PRAGMA integrity_check via rusqlite on destination
    drop(read_conn); // Close the read connection first

    let verify_conn = rusqlite::Connection::open(&dest_path).expect("Failed to open dest db");
    assert_integrity_ok(&verify_conn, "destination after page copy");

    // 8. Verify page count matches
    let dest_verify = SqliteIo::new(&dest_path).unwrap();
    let dest_page_count = dest_verify.page_count().unwrap();
    assert_eq!(
        source_page_count, dest_page_count,
        "Page counts should match"
    );
}

/// Test 2: Multiple complete database copies with integrity checks
///
/// Verifies that copying a complete database multiple times doesn't cause corruption.
/// Note: Partial page writes will corrupt a database - all pages must be written
/// together to maintain consistency.
#[tokio::test]
async fn test_multiple_complete_copies_with_integrity_checks() {
    let tempdir = tempfile::tempdir().unwrap();
    let source_path = tempdir.path().join("multi_source.db");

    // 1. Create a WAL-mode database
    {
        let conn = create_wal_database(&source_path);
        insert_test_data(&conn, 50);
    }

    // Read original pages once
    let source_db = SqliteIo::new(&source_path).unwrap();
    let mut original_pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            original_pages.push((page_num, data.to_vec()));
        })
        .unwrap();
    drop(source_db);

    // 2. Perform multiple complete database copies and verify each
    for copy_num in 0..5 {
        let dest_path = tempdir.path().join(format!("multi_dest_{}.db", copy_num));

        // Write ALL pages in a single transaction (required for consistency)
        let db = AsyncSqliteIo::new(&dest_path).await.unwrap();
        let tx = db
            .transaction(TransactionType::Immediate)
            .await
            .unwrap();

        for (page_num, data) in &original_pages {
            tx.set_page_data(*page_num, data).await.unwrap();
        }

        let _db = tx.commit().await.unwrap();

        // 3. After each complete copy, verify integrity
        let verify_conn =
            rusqlite::Connection::open(&dest_path).expect("Failed to open dest for check");
        assert_integrity_ok(&verify_conn, &format!("copy {}", copy_num));

        // Verify data matches
        let count: i64 = verify_conn
            .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
            .expect("Failed to count");
        assert_eq!(count, 50, "Copy {} should have all rows", copy_num);
    }
}

/// Test 3: WAL Mode Checkpoint Interaction
///
/// Verifies that sqlite_pages works correctly after WAL checkpoint.
#[tokio::test]
async fn test_wal_checkpoint_then_page_write() {
    let tempdir = tempfile::tempdir().unwrap();
    let db_path = tempdir.path().join("checkpoint.db");

    // 1. Create WAL-mode database, write data via rusqlite
    {
        let conn = create_wal_database(&db_path);
        insert_test_data(&conn, 200);

        // 2. Run PRAGMA wal_checkpoint(TRUNCATE) - use query since it returns results
        let _: i64 = conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))
            .expect("Failed to checkpoint");
    }

    // Verify WAL file is truncated (or doesn't exist)
    let wal_path = db_path.with_extension("db-wal");
    if wal_path.exists() {
        let wal_size = std::fs::metadata(&wal_path).unwrap().len();
        // After TRUNCATE checkpoint, WAL should be very small or empty
        assert!(
            wal_size < 4096,
            "WAL file should be small after TRUNCATE checkpoint"
        );
    }

    // 3. Read pages via sqlite_pages
    let source_db = SqliteIo::new(&db_path).unwrap();
    let mut pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            pages.push((page_num, data.to_vec()));
        })
        .unwrap();

    // Write pages to a new database via sqlite_pages
    let dest_path = tempdir.path().join("checkpoint_dest.db");
    let dest_db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = dest_db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    for (page_num, data) in pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }
    let _db = tx.commit().await.unwrap();

    // 4. Verify integrity with rusqlite
    let verify_conn =
        rusqlite::Connection::open(&dest_path).expect("Failed to open dest for verification");
    assert_integrity_ok(&verify_conn, "after checkpoint and page write");

    // Verify data
    let count: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count");
    assert_eq!(count, 200);
}

/// Test 4: Uncommitted Transaction No Corruption
///
/// Verifies that uncommitted sqlite_pages transactions don't corrupt the database.
#[tokio::test]
async fn test_uncommitted_transaction_no_corruption() {
    let tempdir = tempfile::tempdir().unwrap();
    let db_path = tempdir.path().join("uncommitted.db");

    // 1. Create database with initial data
    {
        let conn = create_wal_database(&db_path);
        insert_test_data(&conn, 100);
    }

    // Record initial state
    let initial_conn =
        rusqlite::Connection::open(&db_path).expect("Failed to open for initial check");
    let initial_count: i64 = initial_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count");
    assert_integrity_ok(&initial_conn, "before uncommitted transaction");
    drop(initial_conn);

    // 2. Start sqlite_pages transaction, write some pages
    {
        let db = AsyncSqliteIo::new(&db_path).await.unwrap();
        let tx = db
            .transaction(TransactionType::Immediate)
            .await
            .unwrap();

        // Write some arbitrary data to page 2 (don't commit)
        let junk_data = vec![0xDEu8; 4096];
        tx.set_page_data(2, &junk_data).await.unwrap();

        // 3. DON'T commit - rollback instead
        let _db = tx.rollback().await.unwrap();
    }

    // 4. Open with rusqlite and verify integrity
    let verify_conn =
        rusqlite::Connection::open(&db_path).expect("Failed to open after rollback");
    assert_integrity_ok(&verify_conn, "after rollback");

    // Verify data is unchanged
    let count_after: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count after rollback");
    assert_eq!(initial_count, count_after, "Row count should be unchanged");
}

/// Test 5: Large BLOB Spanning Multiple Pages
///
/// Verifies that writing large data that spans multiple pages works correctly.
#[tokio::test]
async fn test_large_blob_multiple_pages() {
    let tempdir = tempfile::tempdir().unwrap();
    let db_path = tempdir.path().join("large_blob.db");

    // 1. Create database with a table for blobs
    let blob_size = 1024 * 1024; // 1MB blob (creates ~250 pages with 4096 page size)
    let original_blob: Vec<u8> = (0..blob_size).map(|i| (i % 256) as u8).collect();

    {
        let conn = rusqlite::Connection::open(&db_path).expect("Failed to open");
        conn.execute(
            "CREATE TABLE blobs (id INTEGER PRIMARY KEY, data BLOB NOT NULL)",
            [],
        )
        .expect("Failed to create table");

        // 2. Insert a ~1MB blob
        conn.execute("INSERT INTO blobs (data) VALUES (?)", [&original_blob])
            .expect("Failed to insert blob");
    }

    // 3. Read all pages via sqlite_pages
    let source_db = SqliteIo::new(&db_path).unwrap();
    let source_page_count = source_db.page_count().unwrap();
    assert!(
        source_page_count > 200,
        "Should have many pages for 1MB blob, got {}",
        source_page_count
    );

    let mut pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            pages.push((page_num, data.to_vec()));
        })
        .unwrap();

    // 4. Write all pages to a new database via sqlite_pages
    let dest_path = tempdir.path().join("large_blob_dest.db");
    let dest_db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = dest_db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    for (page_num, data) in pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }
    let _db = tx.commit().await.unwrap();

    // 5. Verify both databases pass integrity check
    let source_check_conn =
        rusqlite::Connection::open(&db_path).expect("Failed to open source for check");
    assert_integrity_ok(&source_check_conn, "source database");

    let dest_check_conn =
        rusqlite::Connection::open(&dest_path).expect("Failed to open dest for check");
    assert_integrity_ok(&dest_check_conn, "destination database");

    // 6. Verify blob data matches between databases
    let dest_blob: Vec<u8> = dest_check_conn
        .query_row("SELECT data FROM blobs WHERE id = 1", [], |row| row.get(0))
        .expect("Failed to read dest blob");

    assert_eq!(
        original_blob.len(),
        dest_blob.len(),
        "Blob lengths should match"
    );
    assert_eq!(original_blob, dest_blob, "Blob contents should match");
}

/// Test 6: WAL mode database read by sqlite_pages
///
/// Verifies that sqlite_pages can correctly read a WAL-mode database.
/// Note: When rusqlite closes the connection, it may auto-checkpoint,
/// so we don't assert on WAL file presence.
#[tokio::test]
async fn test_read_wal_database() {
    let tempdir = tempfile::tempdir().unwrap();
    let db_path = tempdir.path().join("wal_read.db");

    // Create WAL-mode database
    {
        let conn = create_wal_database(&db_path);
        insert_test_data(&conn, 100);
    }

    // Read via sqlite_pages - this should work regardless of WAL state
    let db = SqliteIo::new(&db_path).unwrap();
    let page_count = db.page_count().unwrap();
    assert!(page_count >= 2, "Should have pages");

    let mut pages = Vec::new();
    db.page_map(1.., |page_num, data| {
        pages.push((page_num, data.to_vec()));
    })
    .unwrap();

    assert_eq!(pages.len(), page_count, "Should read all pages");
    drop(db);

    // Copy to new database
    let dest_path = tempdir.path().join("wal_read_dest.db");
    let dest_db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = dest_db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    for (page_num, data) in pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }
    let _db = tx.commit().await.unwrap();

    // Verify destination
    let verify_conn =
        rusqlite::Connection::open(&dest_path).expect("Failed to open dest");
    assert_integrity_ok(&verify_conn, "destination from WAL source");

    let count: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count");
    assert_eq!(count, 100, "Should have all rows");
}

/// Test 7: Concurrent sqlite_pages writer and rusqlite reader on same database
///
/// This test simulates the problematic scenario where:
/// - sqlite_pages is writing pages to a database
/// - rusqlite opens the same database for reading (e.g., startup conditions check)
#[tokio::test]
async fn test_concurrent_sqlite_pages_write_rusqlite_read_same_db() {
    let tempdir = tempfile::tempdir().unwrap();
    let db_path = tempdir.path().join("concurrent_same.db");

    // Create initial database
    {
        let conn = create_wal_database(&db_path);
        insert_test_data(&conn, 50);
    }

    // Read source pages
    let source_db = SqliteIo::new(&db_path).unwrap();
    let mut pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            pages.push((page_num, data.to_vec()));
        })
        .unwrap();
    drop(source_db);

    // Create destination database with some initial structure
    let dest_path = tempdir.path().join("concurrent_dest.db");
    {
        let conn = create_wal_database(&dest_path);
        insert_test_data(&conn, 10); // Some initial data
    }

    // Open sqlite_pages for writing
    let dest_db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = dest_db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    // While transaction is open, try to open with rusqlite for reading
    // This simulates the startup_conditions check scenario
    let read_result = rusqlite::Connection::open_with_flags(
        &dest_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    );

    // The read connection might succeed or fail depending on locking
    // What matters is that if it succeeds, it shouldn't cause corruption
    if let Ok(read_conn) = read_result {
        // Try to read while sqlite_pages has the transaction open
        let _count_result: Result<i64, _> =
            read_conn.query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0));
        // Don't assert on result - it may or may not work depending on lock state
    }

    // Write pages
    for (page_num, data) in pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }

    // Commit
    let _db = tx.commit().await.unwrap();

    // After commit, verify integrity
    let verify_conn =
        rusqlite::Connection::open(&dest_path).expect("Failed to open for verification");
    assert_integrity_ok(&verify_conn, "after concurrent access scenario");
}

/// Test 8: Simulate replication pattern - incremental page writes
///
/// This simulates the actual data_plane usage pattern where pages come
/// in batches from a replication stream, with multiple commits.
#[tokio::test]
async fn test_replication_pattern_incremental_writes() {
    let tempdir = tempfile::tempdir().unwrap();
    let source_path = tempdir.path().join("replication_source.db");
    let dest_path = tempdir.path().join("replication_dest.db");

    // Create source with substantial data
    {
        let conn = create_wal_database(&source_path);
        insert_test_data(&conn, 500);

        // Add some variety - create additional tables
        conn.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO metadata VALUES ('version', '1.0'), ('schema', 'v2')",
            [],
        )
        .unwrap();

        // Checkpoint to ensure clean state - use query since it returns results
        let _: i64 = conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))
            .unwrap();
    }

    // Read all pages from source
    let source_db = SqliteIo::new(&source_path).unwrap();
    let total_pages = source_db.page_count().unwrap();
    let mut all_pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            all_pages.push((page_num, data.to_vec()));
        })
        .unwrap();
    drop(source_db);

    // Simulate batched replication - write in chunks like data_plane does
    let batch_size = 10;
    let num_batches = (all_pages.len() + batch_size - 1) / batch_size;

    for batch_num in 0..num_batches {
        let start = batch_num * batch_size;
        let end = ((batch_num + 1) * batch_size).min(all_pages.len());
        let batch = &all_pages[start..end];

        // Open, write batch, commit - like data_plane does per transaction
        let db = AsyncSqliteIo::new(&dest_path).await.unwrap();
        let tx = db
            .transaction(TransactionType::Immediate)
            .await
            .unwrap();

        for (page_num, data) in batch {
            tx.set_page_data(*page_num, data).await.unwrap();
        }

        let _db = tx.commit().await.unwrap();
    }

    // Final verification
    let verify_conn = rusqlite::Connection::open(&dest_path).expect("Failed to open dest");
    assert_integrity_ok(&verify_conn, "after replication pattern");

    // Verify page count matches
    let dest_db = SqliteIo::new(&dest_path).unwrap();
    let dest_page_count = dest_db.page_count().unwrap();
    assert_eq!(
        total_pages, dest_page_count,
        "Page counts should match after replication"
    );

    // Verify data
    let notes_count: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count notes");
    assert_eq!(notes_count, 500);

    let version: String = verify_conn
        .query_row(
            "SELECT value FROM metadata WHERE key = 'version'",
            [],
            |row| row.get(0),
        )
        .expect("Failed to get version");
    assert_eq!(version, "1.0");
}

/// Test 9: Page write order independence
///
/// Verifies that pages can be written in any order and still produce
/// a valid database.
#[tokio::test]
async fn test_page_write_order_independence() {
    let tempdir = tempfile::tempdir().unwrap();
    let source_path = tempdir.path().join("order_source.db");

    // Create source
    {
        let conn = create_wal_database(&source_path);
        insert_test_data(&conn, 100);
    }

    // Read pages
    let source_db = SqliteIo::new(&source_path).unwrap();
    let mut pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            pages.push((page_num, data.to_vec()));
        })
        .unwrap();
    drop(source_db);

    // Reverse the order of pages (write high-numbered pages first)
    let mut reversed_pages = pages.clone();
    reversed_pages.reverse();

    // Write in reversed order
    let dest_path = tempdir.path().join("order_dest.db");
    let db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    for (page_num, data) in reversed_pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }
    let _db = tx.commit().await.unwrap();

    // Verify
    let verify_conn = rusqlite::Connection::open(&dest_path).expect("Failed to open dest");
    assert_integrity_ok(&verify_conn, "after reversed page write");

    let count: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .expect("Failed to count");
    assert_eq!(count, 100);
}

/// Test 10: Verify header page (page 1) special handling
///
/// The header page contains database metadata. Verify it's handled correctly.
#[tokio::test]
async fn test_header_page_handling() {
    let tempdir = tempfile::tempdir().unwrap();
    let source_path = tempdir.path().join("header_source.db");

    // Create source with specific page size
    {
        let conn = rusqlite::Connection::open(&source_path).expect("Failed to open");
        // Set page size before creating any tables
        conn.pragma_update(None, "page_size", 4096).unwrap();
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)", [])
            .unwrap();

        for i in 0..50 {
            conn.execute(
                "INSERT INTO test (data) VALUES (?)",
                [format!("data_{}", i)],
            )
            .unwrap();
        }
    }

    // Read pages
    let source_db = SqliteIo::new(&source_path).unwrap();
    let mut pages = Vec::new();
    source_db
        .page_map(1.., |page_num, data| {
            pages.push((page_num, data.to_vec()));
        })
        .unwrap();

    // Verify page 1 exists and is 4096 bytes
    let page1 = pages.iter().find(|(n, _)| *n == 1).expect("Page 1 not found");
    assert_eq!(page1.1.len(), 4096, "Page 1 should be 4096 bytes");

    // First 16 bytes of page 1 should contain SQLite header magic
    let header = &page1.1[0..16];
    assert_eq!(
        &header[0..6],
        b"SQLite",
        "Header should start with 'SQLite'"
    );

    drop(source_db);

    // Copy to destination
    let dest_path = tempdir.path().join("header_dest.db");
    let dest_db = AsyncSqliteIo::new(&dest_path).await.unwrap();
    let tx = dest_db
        .transaction(TransactionType::Immediate)
        .await
        .unwrap();

    for (page_num, data) in pages {
        tx.set_page_data(page_num, &data).await.unwrap();
    }
    let _db = tx.commit().await.unwrap();

    // Verify destination
    let verify_conn = rusqlite::Connection::open(&dest_path).expect("Failed to open dest");
    assert_integrity_ok(&verify_conn, "after header page copy");

    // Verify page size is preserved
    let dest_page_size: i64 = verify_conn
        .pragma_query_value(None, "page_size", |row| row.get(0))
        .expect("Failed to get page size");
    assert_eq!(dest_page_size, 4096);

    // Verify data
    let count: i64 = verify_conn
        .query_row("SELECT COUNT(*) FROM test", [], |row| row.get(0))
        .expect("Failed to count");
    assert_eq!(count, 50);
}
