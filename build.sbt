name := "dl4j-spark-sbt-sample"

version := "1.0"

scalaVersion := "2.11.7"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.0",
  "org.apache.spark" %% "spark-mllib" % "1.5.0",
  "org.deeplearning4j" % "dl4j-spark" % "0.4-rc3.9" exclude("org.apache.spark", "spark-core_2.10"),
  "org.nd4j" % "nd4j-native" % "0.4-rc3.9" classifier "" classifier "linux-x86_64",
  "org.nd4j" % "nd4j-api" % "0.4-rc3.9"
)

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)
