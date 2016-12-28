package eucalyptus.decisiontree

import scala.language.implicitConversions
import scala.util.Random
import scala.util.control.Breaks._

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
  var tree: Option[BiTree] = None
  var predictors: Option[List[String]] = None
  var response: Option[String] = None
  var weight: Option[String] = None
  var maxFeaturesPerSplit: Option[Int] = None

  protected type FeatureSupportDict = Option[Map[String, Option[SupportDict]]]

  protected case class SupportDict(
      val bins: Option[Vector[NumericalValue]] = None, val maxRefined: Option[Boolean] = None,
      val catMap: Option[Map[DataValue, NumericalValue]] = None,
      val catMappedData: Option[Series[NumericalValue]] = None)

  def fit(
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {
    var parsedX: DataFrame = (y, predictors, response) match {
      case (None, Some(predictors), Some(response)) => {
        // Perform basic checks
        if (! predictors.toSet.subsetOf(x.getColumns.toSet))
          throw new RuntimeException("Invalid preditors argument!")
        if (! x.getColumns.toSet.contains(response))
          throw new RuntimeException("Invalid response argument!")

        this.predictorColNames = Some(predictors)
        this.responseColName = Some(response)

        x.select(predictors :+ response)
      }
      // case (None, Some(predictors), None) => {}
      // case (None, None, Some(response)) => {}
      // case (None, None, None) => {}
      // case (Some(y), Some(predictors), None) => {}
      // case (Some(y), None, None) => {}
      case _ => throw new RuntimeException("Improper arguments passed to fit method!")
    }

    weightColName = Some(getDistinctColumnName(this.predictorColNames.get :+ this.responseColName.get, "weight"))

    parsedX = weights match {
      case Some(weights) => parsedX.update(weightColName.get, weights)
      case None => parsedX.update(weightColName.get, NumericalValue(1))
    }

    maxFeaturesPerSplit = Some(getMaxFeaturesPerSplit)
    // fitRecursive(parsedX, 0)
  }
  // def predict[T](x: DataFrame): T
  // def predict[T](x: Row): T
  private def fitRecursive(
      data: DataFrame, depth: Int, support: FeatureSupportDict = None,
      parent: Option[BiNode] = None, key: Option[String] = None): Unit = {
    val mySupport = support.getOrElse(
      predictorColNames.map(predictor => (predictor -> (None: Option[SupportDict]))).toMap)
    var costImprovement: NumericalValue = NumericalValue(0)
    var feature: Option[String] = None
    var split: Option[NumericalValue] = None
    var leftData: Option[DataFrame] = None
    var rightData: Option[DataFrame] = None

    // Find optional split, if enough samples in parent node to split and max depth not exceeded.
    if (data[NumericalValue](weightColName).sum >= minSamplesSplit && depth < maxDepth) {
      var numFeaturesConsidered: Int = 0
      breakable {
        for (feature <- Random.shuffle(predictorColNames)) {

        }
      }

      // Create child node or leaf (figure out by context)
      val child: Node = if (costImprovement > 0) {

      } else {}

      // Mount child in tree

      // Fit deeper branches if current child node is not a leaf
      if ()! child.isLeaf) {
        fitRecursive(leftData, depth+1, support?,, child, false)
        fitRecursive(rightData, depth+1, support?,, child, true)
      }
    }

  }

  private def preprocessData(): SupportDict = {}
  private def findBestSplit(): Tuple


  protected def getMaxFeaturesPerSplit: Int = predictors.get.length

  private def getDistinctColumnName(columnNames: List[String], newNameBase: String): String = {
    var newName = newNameBase
    var i = 0

    while (columnNames contains newName) {
      newName =  newNameBase + i.toString
      i += 1
    }

    newName
  }
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
