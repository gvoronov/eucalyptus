package eucalyptus.decisiontree

import scala.language.implicitConversions

import koalas.dataframe.DataFrame
import koalas.series.Series
import koalas.row.Row
import koalas.datavalue._
import koalas.numericalops.NumericalOps._

import eucalyptus.tree._

object DecisionTree {
  implicit def wrapToOption[T](x: T) = Option[T](x)
}

abstract class DecisionTree(
    val maxSplitPoints: Int = 10, val minSplitPoints: Int = 1, val maxDepth: Int = 100,
    val minSamplesSplit: Int = 2, val minSamplesLeaf: Int = 1) {
  // var predictors:  List[String]
  // var response: String
  def fit(
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {
    val parsedX: DataFrame = (y, predictors, response) match {
      case (None, Some(predictors), Some(response)) => {

      }
      case _ => throw new RuntimeException("")
    }
    if (y.isDefined && predictors.isDefined && response.isDefined) {

    } else if
    println(response.getOrElse("asdf"))
  }
  // def predict[T](x: DataFrame): T
  // def predict[T](x: Row): T
  // private def fitRecursive(): Unit
  // private def findBestSplit(): Tuple
}

trait RegressionTreeLike {
  protected def summarizeResponse = {}
  protected def evalCostFromBlock = {}
  protected def EvalResponseOnCat = {}
  protected def reduceBlockSummary = {}
}

trait ClassificationTreeLike
trait BivariateClassificationTreeLike extends ClassificationTreeLike
trait MultivariateClassificationTreeLike extends ClassificationTreeLike
trait DecorrelatedTreeLike

class RegressionTree extends DecisionTree with RegressionTreeLike

class DecorrelatedRegressionTree extends DecisionTree with RegressionTreeLike with DecorrelatedTreeLike
