CREATE DOMAIN UInt256 AS NUMERIC(78, 0) 
CHECK (VALUE >= 0 AND VALUE < 2^256);