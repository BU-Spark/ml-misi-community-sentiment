ALTER TABLE users
  ADD COLUMN is_guest BOOLEAN NOT NULL DEFAULT FALSE AFTER profile_complete;

CREATE INDEX idx_users_is_guest_created ON users (is_guest, created_at);
