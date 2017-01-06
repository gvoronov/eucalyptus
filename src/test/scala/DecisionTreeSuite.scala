package eucalyptus.test

import org.scalatest.FunSuite
import org.scalatest.Assertions._

import koalas.datasets.pricing.Pricing
// import koalas.row.Row
// import koalas.datavalue._
// import koalas.numericalops.NumericalOps._

// import eucalyptus.tree._
import eucalyptus.decisiontree._
import eucalyptus.decisiontree.DecisionTree._

class DecisionTreeSuite extends FunSuite {
  test("fit tree on pricing data") {
    val df = Pricing.makeDF()

    val predictors = List("p_1", "p_2", "p_3", "p_4")
    val response = "q_1"

    val rtree = new RegressionTree
    rtree.fit(x = df, predictors = predictors, response = response)
  }

}
