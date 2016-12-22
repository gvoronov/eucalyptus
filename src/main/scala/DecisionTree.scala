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
  var predictors: Option[List[String]] = None
  var response: Option[String] = None

  def fit(
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {
    val parsedX: DataFrame = (y, predictors, response) match {
      case (None, Some(predictors), Some(response)) => {
        // Perform basic checks
        if (! predictors.toSet.subsetOf(x.getColumns.toSet))
          throw new RuntimeException("Invalid preditors argument!")
        if (! x.getColumns.toSet.contains(response))
          throw new RuntimeException("Invalid response argument!")

        this.predictors = Some(predictors)
        this.response = Some(response)

        x  //selected down to predictors and response
      }
      // case (None, Some(predictors), None) => {}
      // case (None, None, Some(response)) => {}
      // case (None, None, None) => {}
      // case (Some(y), Some(predictors), None) => {}
      // case (Some(y), None, None) => {}
      case _ => throw new RuntimeException("Improper arguments passed to fit method!")
    }

  // val weightCol =
  // parsedX.update(weightCol, weights.getOrElse())
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
