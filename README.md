## dl4j-spark-sbt-example

This is an example project(on linux-x86_64) of DL4J on Spark with sbt, not using `dl4j-spark-ml`. You can run this with only sbt(without spark-submit).

If you run this on OS other than linux-x86_64, need adjust a parameter of `classifier` in `build.sbt`.

### How to run

```
sbt run
```

and then choose one example.

### Others

* If you get OutOfMemoryError, run `sbt` with `-mem` option
* These examples are following [dl4j-spark-cdh5-examples](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples)
