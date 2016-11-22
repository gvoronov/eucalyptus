package eucalyptus.util

import scala.math._
import scala.collection.mutable.ArraySeq

import breeze.stats.sampling.standardBasis._

import koalas.series.Series
import koalas.numericalops.NumericalOps._
import koalas.datavalue._

object TreeUtil{
  /**
   * Generate an equidepth histogram for input Series[NumericalValue]
   * @param data
   * @param numBins
   * @param removeDups
   */
  def equidepthHist(data: Series[NumericalValue], numBins: Int, removeDups: Boolean = true):
      Vector[NumericalValue] = {
    // Remove any nan values
    val myData = data.filter(! _.isNaN)

    // Calculate index speration to use in order to bin sorted array. Typically will have remainder
    // number of elements which will not evenly be divided by bins. Randomly chose which bins will
    // get an extra element.
    val myNumBins: Int = min(numBins, myData.length)

    if(myNumBins == 0)
      return Vector.empty[NumericalValue]

    val binWidths: ArraySeq[Int] = ArraySeq.fill(myNumBins)(myData.length / myNumBins)
    for (i <- subsetsOfSize(0 until myNumBins, myData.length % myNumBins).draw)
      binWidths(i) += 1

    // Sort input data
    val sortedData = myData.sorted

    // Calculate bin separation points.
    val bins: ArraySeq[NumericalValue] = ArraySeq.fill(myNumBins + 1)(null)
    var position: Int = 0
    bins(position) = sortedData(position)
    for ((binWidth, i) <- binWidths.zipWithIndex) {
      position += binWidth
      bins(i + 1) = sortedData(position)
    }
    bins(myNumBins) = sortedData.last

    // Remove zero width bins if any present and flag set to do so.
    if (removeDups)
      bins.distinct.toVector
    else
      bins.toVector
  }
}
