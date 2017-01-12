package eucalyptus.ensemble

import scala.util.Random

import koalas.dataframe.DataFrame
import koalas.series.Series
import koalas.row.Row
import koalas.datavalue._
import koalas.numericalops.NumericalOps._

import eucalyptus.decisiontree._

sealed trait Bagging {
  val forest: List[DecisionTree]
  def fit (
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {
    val numObs = x.length
    def fitTree(tree: DecisionTree): Unit = {
      val replicationIndex = Vector.fill(numObs)(Random.nextInt(numObs))
      val replicationX = x(replicationIndex)
      val replicationY = y.map(_(replicationIndex))
      val replicaionWeights = weights.map(_(replicationIndex))
      tree.fit(replicationX, replicationY, predictors, response, replicaionWeights)
    }
    forest.foreach(fitTree)
  }

  def predictEnsemble[T](x: Row): Series[T] = Series[T](forest.map(_.predict(x).asInstanceOf[T]))
  def predictEnsemble[T](x: DataFrame): Series[Series[T]] = x.map(predictEnsemble(_))
}

class BaggingRegressor(
    val numTrees: Int = 10, val maxSplitPoints: Int = 10, val minSplitPoints: Int = 1,
    val maxDepth: Int = 100, val minSamplesSplit: Int = 2, val minSamplesLeaf: Int = 1)
    extends Bagging {
  val forest: List[DecisionTree] = List.fill(numTrees)(
    new RegressionTree(maxSplitPoints, minSplitPoints, maxDepth, minSamplesSplit, minSamplesLeaf))

  def predict(x: DataFrame): Series[NumericalValue] = predictEnsembleMean(x)
  def predict(x: Row): NumericalValue = predictEnsembleMean(x)
  def predictEnsembleMean(x: DataFrame): Series[NumericalValue] = predictEnsemble[NumericalValue](x).map(_.mean)
  def predictEnsembleMean(x: Row): NumericalValue = predictEnsemble[NumericalValue](x).mean
  def predictEnsembleVar(x: DataFrame): Series[NumericalValue] = predictEnsemble[NumericalValue](x).map(_.variance)
  def predictEnsembleVar(x: Row): NumericalValue = predictEnsemble[NumericalValue](x).variance
}
// class BaggingClassifier extends Bagging
// class BaggingBivariateClassifier extends BaggingClassifier
// class BaggingMultivariateClassifier extends BaggingClassifier
class RandomForestRegressor(
    numTrees: Int = 10, maxSplitPoints: Int = 10, minSplitPoints: Int = 1, maxDepth: Int = 100,
    minSamplesSplit: Int = 2, minSamplesLeaf: Int = 1, val maxFeaturesPerSplit: Any = "sqrt")
    extends BaggingRegressor(
      numTrees, maxSplitPoints, minSplitPoints, maxDepth, minSamplesSplit, minSamplesLeaf) {
  override val forest: List[DecisionTree] = List.fill(numTrees)(
    new DecorrelatedRegressionTree(
      maxSplitPoints, minSplitPoints, maxDepth, minSamplesSplit, minSamplesLeaf,
      maxFeaturesPerSplit))
}
// class RandomForestClassifier extends BaggingClassifier
// class RandomForestBivariateClassifier extends RandomForestClassifier
// class RandomForestMultivariateClassifier extends RandomForestClassifier
// trait Boosting
// trait AdaBoost
// trait GradientBoosting
