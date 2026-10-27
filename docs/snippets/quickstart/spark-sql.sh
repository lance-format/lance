spark-sql \
    --packages org.lance:lance-spark-bundle-4.2_2.13:0.8.0 \
    --conf spark.sql.extensions=org.lance.spark.extensions.LanceSparkSessionExtensions \
    --conf spark.sql.catalog.lance=org.lance.spark.LanceNamespaceSparkCatalog \
    --conf spark.sql.catalog.lance.impl=dir \
    --conf spark.sql.catalog.lance.root=/tmp/lance-spark-quickstart \
    --conf spark.sql.defaultCatalog=lance \
    --conf spark.hadoop.hive.cli.print.header=true
