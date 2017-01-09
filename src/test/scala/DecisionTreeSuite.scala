package eucalyptus.test

import java.time._

import scala.util.Random

import org.scalatest.FunSuite
import org.scalatest.Assertions._

import koalas.datasets.pricing.Pricing
import koalas.datavalue._
import koalas.numericalops.NumericalOps._

import eucalyptus.decisiontree._
import eucalyptus.decisiontree.DecisionTree._

class DecisionTreeWPricingSuite extends FunSuite {
  val df = Pricing.makeDF()
  val (trainingDf, testingDf) = df.partition(row => Random.nextDouble < 0.9)

  test("no categorical predictors") {
    val predictors = List("p_0", "p_1", "p_2", "p_3", "p_4")
    val response = "q_1"

    val rtree = new RegressionTree(maxDepth = Int.MaxValue)

    var t = Instant.now
    rtree.fit(x = trainingDf, predictors = predictors, response = response)
    val fitTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    t = Instant.now
    val predictions = rtree.predict[NumericalValue](testingDf)
    val predictTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    val mse = ((predictions :- testingDf[NumericalValue]("q_1")) :** 2).mean
    val r2 = NumericalValue(1) - (mse / testingDf[NumericalValue]("q_1").variance)

    info("Fit Time: " + fitTime.toString)
    info("Predict Time: " + predictTime.toString)
    info("Coefficient of Determination (R**2): " + r2.toString)
  }

  test("only week categorical predictors") {
    val predictors = List("week", "p_0", "p_1", "p_2", "p_3", "p_4")
    val response = "q_1"

    val rtree = new RegressionTree(maxDepth = Int.MaxValue)

    var t = Instant.now
    rtree.fit(x = trainingDf, predictors = predictors, response = response)
    val fitTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    t = Instant.now
    val predictions = rtree.predict[NumericalValue](testingDf)
    val predictTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    val mse = ((predictions :- testingDf[NumericalValue]("q_1")) :** 2).mean
    val r2 = NumericalValue(1) - (mse / testingDf[NumericalValue]("q_1").variance)

    info("Fit Time: " + fitTime.toString)
    info("Predict Time: " + predictTime.toString)
    info("Coefficient of Determination (R**2): " + r2.toString)
  }

  test("only store categorical predictors") {
    val predictors = List("store", "p_0", "p_1", "p_2", "p_3", "p_4")
    val response = "q_1"

    val rtree = new RegressionTree(maxDepth = Int.MaxValue)

    var t = Instant.now
    rtree.fit(x = trainingDf, predictors = predictors, response = response)
    val fitTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    t = Instant.now
    val predictions = rtree.predict[NumericalValue](testingDf)
    val predictTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    val mse = ((predictions :- testingDf[NumericalValue]("q_1")) :** 2).mean
    val r2 = NumericalValue(1) - (mse / testingDf[NumericalValue]("q_1").variance)

    info("Fit Time: " + fitTime.toString)
    info("Predict Time: " + predictTime.toString)
    info("Coefficient of Determination (R**2): " + r2.toString)
  }

  test("both store and week categorical predictors") {
    val predictors = List("store", "week", "p_0", "p_1", "p_2", "p_3", "p_4")
    val response = "q_1"

    val rtree = new RegressionTree(maxDepth = Int.MaxValue)

    var t = Instant.now
    rtree.fit(x = trainingDf, predictors = predictors, response = response)
    val fitTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    t = Instant.now
    val predictions = rtree.predict[NumericalValue](testingDf)
    val predictTime = Duration.between(t, Instant.now).toMillis.toDouble / 1000

    val mse = ((predictions :- testingDf[NumericalValue]("q_1")) :** 2).mean
    val r2 = NumericalValue(1) - (mse / testingDf[NumericalValue]("q_1").variance)

    info("Fit Time: " + fitTime.toString)
    info("Predict Time: " + predictTime.toString)
    info("Coefficient of Determination (R**2): " + r2.toString)
  }
}
