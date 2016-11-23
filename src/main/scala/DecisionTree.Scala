package eucalyptus.decisiontree

import koalas.dataframe.DataFrame
import koalas.series.Series
import koalas.row.Row
import koalas.datavalue._
import koalas.numericalops.NumericalOpsImps._

import eucalyptus.tree._

abstract class DecisionTree(
    val maxSplitPoints: Int = 10, val minSplitPoints: Int = 1, val maxDepth: Int = 100,
    val minSamplesSplit: Int = 2, val minSamplesLeaf: Int = 1) {
  def fit(x: DataFrame, predictors: List[String], response: String, weightArray: Series[NumericalValue]): Unit
  def predict[T](x: DataFrame): T
  def predict[T](x: Row): T
  private def fitRecursive(): Unit
  private def findBestSplit(): Tuple
}

trait RegressionTreeLike {
  protected def summarizeResponse
  protected def evalCostFromBlock
  protected def EvalResponseOnCat
  protected def reduceBlockSummary
}

trait ClassificationTreeLike
trait BivariateClassificationTreeLike extends ClassificationTreeLike
trait MultivariateClassificationTreeLike extends ClassificationTreeLike
trait DecorrelatedTreeLike

class RegressionTree extends DecisionTree with RegressionTreeLike

class DecorrelatedRegressionTree extends DecisionTree with RegressionTreeLike with DecorrelatedTreeLike
