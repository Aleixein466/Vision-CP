"""Transactional local records; legacy JSONL stays intact as migration evidence."""
import json
import sqlite3
import threading
from contextlib import contextmanager, closing
from datetime import datetime, timezone
from pathlib import Path


class DatasetStore:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'studio.sqlite3'
        self.local = threading.local()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(kind,id))''')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS migration_errors (file TEXT,line INTEGER,reason TEXT)')
            if not db.execute("SELECT 1 FROM metadata WHERE key='migrated'").fetchone():
                for kind, filename, key in [('candidate','candidates.jsonl','id'), ('annotation','manifest.jsonl','annotation_id')]:
                    path = self.root / filename
                    if not path.exists():
                        continue
                    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            identity = row.get(key) or (f'legacy-{number}' if kind == 'annotation' else None)
                            if not identity or not isinstance(row, dict):
                                raise ValueError('missing identifier')
                            if db.execute('SELECT 1 FROM records WHERE kind=? AND id=?', (kind, identity)).fetchone():
                                raise ValueError('duplicate identifier; original retained in JSONL')
                            db.execute('INSERT INTO records VALUES (?,?,?)', (kind, identity, json.dumps(row)))
                        except (ValueError, TypeError, AttributeError) as exc:
                            db.execute('INSERT INTO migration_errors VALUES (?,?,?)', (filename, number, str(exc)))
                db.execute("INSERT INTO metadata VALUES ('migrated',?)", (datetime.now(timezone.utc).isoformat(),))
        self.backup('migration')

    @contextmanager
    def transaction(self):
        existing = getattr(self.local, 'db', None)
        if existing is not None:
            yield existing
            return
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        self.local.db = db
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            self.local.db = None
            db.close()

    def records(self, kind):
        with self.transaction() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM records WHERE kind=? ORDER BY rowid', (kind,))]

    def put(self, kind, records):
        key = 'id' if kind == 'candidate' else 'annotation_id'
        with self.transaction() as db:
            for row in records:
                db.execute('INSERT INTO records VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload',
                           (kind, row[key], json.dumps(row, ensure_ascii=False, allow_nan=False)))

    def errors(self):
        with self.transaction() as db:
            return [dict(zip(('file', 'line', 'reason'), row)) for row in db.execute('SELECT * FROM migration_errors')]

    def backup(self, name=None):
        folder = self.root / 'backups'
        folder.mkdir(exist_ok=True)
        path = folder / ((name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')) + '.sqlite3')
        if not path.exists():
            with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(path)) as dest:
                source.backup(dest)
        return path


def transactional(method):
    def wrapped(self, *args, **kwargs):
        with self.lock, self.store.transaction():
            return method(self, *args, **kwargs)
    return wrapped
