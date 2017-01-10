package eucalyptus.ensemble

class EnsembleModel

trait Bagging extends EnsembleModel {
  def fit (
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {

  }
  def predictEnsemble(x)

}

class BaggingRegressor extends Bagging
class BaggingClassifier extends Bagging {
  throw new RuntimeException("Not implemented yet!")
}
class BaggingBivariateClassifier extends BaggingClassifier {
  throw new RuntimeException("Not implemented yet!")
}
class BaggingMultivariateClassifier extends BaggingClassifier {
  throw new RuntimeException("Not implemented yet!")
}
trait RandomForest
class RandomForestRegressor  extends BaggingRegressor with RandomForest {

}
class RandomForestClassifier extends BaggingClassifier with RandomForest {
  throw new RuntimeException("Not implemented yet!")
}
class RandomForestBivariateClassifier extends RandomForestClassifier {
  throw new RuntimeException("Not implemented yet!")
}
class RandomForestMultivariateClassifier extends RandomForestClassifier {
  throw new RuntimeException("Not implemented yet!")
}
trait Boosting
trait AdaBoost
trait GradientBoosting
