package eucalyptus.test

import scala.util.Random

import org.scalatest.FunSuite
import org.scalatest.Assertions._

import koalas.datasets.pricing.Pricing
// import koalas.row.Row
import koalas.datavalue._
import koalas.numericalops.NumericalOps._

// import eucalyptus.tree._
import eucalyptus.decisiontree._
import eucalyptus.decisiontree.DecisionTree._

class DecisionTreeSuite extends FunSuite {
  test("test w/ pricing data") {
    val df = Pricing.makeDF(weekWeight = 0.0, storeWeight = 0.0)
    // val df = Pricing.makeDF()
    val (trainingDf, testingDf) = df.partition(row => Random.nextDouble < 0.9)

    val predictors = List("p_0", "p_1", "p_2", "p_3", "p_4")
    val response = "q_1"

    val rtree = new RegressionTree(maxDepth = Int.MaxValue)
    rtree.fit(x = trainingDf, predictors = predictors, response = response)

    val mse = ((rtree.predict[NumericalValue](testingDf) :- testingDf[NumericalValue]("q_1")) :** 2).mean
    val r2 = NumericalValue(1) - (mse / testingDf[NumericalValue]("q_1").variance)
    info(r2.toString)
  }

}
