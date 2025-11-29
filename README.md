# sqlite_pages

Page-level SQLite database access using the [`sqlite_dbpage`](https://sqlite.org/dbpage.html) virtual table.

## Overview

This library provides direct access to SQLite's raw database pages via the `SQLITE_DBPAGE` virtual table. This virtual table exposes the underlying database file at the page level, allowing reads and writes of raw binary page content through SQLite's pager layer.

This is **not** for normal database operations - use standard SQLite queries for reading and writing data.

## Build Configuration

Add the following to your `.cargo/config.toml`:

```toml
[env]
LIBSQLITE3_FLAGS = "-DSQLITE_ENABLE_DBPAGE_VTAB"
```

Without this flag, operations will fail with "no such table: sqlite_dbpage".

## Usage

### Reading Pages

```rust
use sqlite_pages::SqliteIo;

let db = SqliteIo::new("database.db")?;

// Iterate over all pages
db.page_map(.., |page_num, data| {
    println!("Page {}: {} bytes", page_num, data.len());
})?;
```

### Writing Pages

```rust
use sqlite_pages::{SqliteIo, TransactionType};

let db = SqliteIo::new("target.db")?;
let mut tx = db.transaction(TransactionType::Immediate)?;

for (page_num, data) in pages {
    tx.set_page_data(page_num, &data)?;
}

tx.commit()?;
```

### Async API

```rust
use sqlite_pages::{AsyncSqliteIo, TransactionType};

let db = AsyncSqliteIo::new("target.db").await?;
let tx = db.transaction(TransactionType::Immediate).await?;

tx.set_page_data(1, &page_data).await?;
tx.commit().await?;
```

## Important Notes

- **Page Numbering**: All page numbers are **1-based** (the first page is page 1, not page 0)
- **Database Corruption Risk**: This API bypasses SQLite's safety mechanisms. Writing invalid page data will corrupt your database. Only use this for copying valid pages from another SQLite database.

## Transaction Types

- `Deferred`: No lock acquired until first read/write operation
- `Immediate`: Write lock acquired immediately (recommended for writes)
- `Exclusive`: Exclusive lock acquired immediately

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
