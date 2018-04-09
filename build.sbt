name := "mpc-bn"

version := "1.0"

scalaVersion := "2.12.2"

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies ++= Seq (
  "colt" % "colt" % "1.2.0",
  "org.moeaframework" % "moeaframework" % "2.12",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.apache.commons" % "commons-lang3" % "3.6",
  "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.1",
  "com.google.code.gson" % "gson" % "2.8.2"
)
    