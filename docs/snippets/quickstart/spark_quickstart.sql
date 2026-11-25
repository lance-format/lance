-- Runnable source for the SQL tabs in docs/src/quickstart/spark.md. Sections
-- between the `--8<--` markers are embedded in the page. CI runs the file with
-- the command in spark-sql.sh and checks each section's results against
-- outputs/ (see tests/snippet_runner.py).

-- --8<-- [start:create]
CREATE TABLE products (
    id BIGINT NOT NULL,
    name STRING,
    category STRING,
    price DOUBLE,
    vector ARRAY<FLOAT> NOT NULL
) TBLPROPERTIES ('vector.arrow.fixed-size-list.size' = '4');

INSERT INTO products VALUES
    (1, 'red running shoes', 'shoes', 89.0, array(1.0, 0.0, 0.0, 0.25)),
    (2, 'blue trail running shoes', 'shoes', 120.0, array(0.875, 0.125, 0.0, 0.375)),
    (3, 'leather hiking boots', 'shoes', 150.0, array(0.625, 0.0, 0.5, 0.25)),
    (4, 'waterproof rain jacket', 'outerwear', 99.0, array(0.0, 1.0, 0.25, 0.0)),
    (5, 'wool winter coat', 'outerwear', 210.0, array(0.25, 0.75, 0.625, 0.0));
-- --8<-- [end:create]

-- --8<-- [start:query]
SELECT id, name, price FROM products WHERE price < 100 ORDER BY id;

SELECT category, count(*) AS num_products, avg(price) AS avg_price
FROM products
GROUP BY category
ORDER BY category;
-- --8<-- [end:query]

-- --8<-- [start:vector-search]
SELECT id, name, _distance
FROM VECTOR_SEARCH(
    table => 'products',
    query_vector => array(0.875, 0.125, 0.0, 0.25),
    num_results => 3
)
ORDER BY _distance;
-- --8<-- [end:vector-search]

-- --8<-- [start:fts]
ALTER TABLE products CREATE INDEX name_idx USING fts (name);

SELECT id, name FROM products WHERE lance_match(name, 'running') ORDER BY id;
-- --8<-- [end:fts]

-- --8<-- [start:modify]
INSERT INTO products VALUES
    (6, 'trail socks', 'accessories', 15.0, array(0.75, 0.25, 0.25, 0.25));
UPDATE products SET price = 79.0 WHERE id = 1;
DELETE FROM products WHERE category = 'outerwear';

SELECT id, name, price FROM products ORDER BY id;
-- --8<-- [end:modify]
