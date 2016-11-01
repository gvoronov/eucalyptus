package eucalyptus.test

import org.scalatest.FunSuite

import koalas.datasets.bikesharing.BikeSharing

import eucalyptus.tree._

class TreeSuite extends FunSuite {
  val df = BikeSharing.loadDayDf

  def buildSmallTree: BiTree = {
    val tree = new BiTree(BiNode("atemp", 500))

    val gen0 = tree.root.setAndGetChild(false, new RegressionLeaf())
    val gen1 = tree.root.setAndGetChild(true, new BiNode("holiday"))

    val gen10 = gen1.setAndGetChild(false, new RegressionLeaf)
    val gen11 = gen1.setAndGetChild(true, new RegressionLeaf)

    tree
  }

  def buildMediumTree: BiTree = {
    val tree = new BiTree(BiNode("cnt", 500))

    val gen0 = tree.root.setAndGetChild(false)
    val gen1 = tree.root.setAndGetChild(true)

    val gen00 = gen1.setAndGetChild(false)
    val gen01 = gen1.setAndGetChild(true)
    // Look at a simple cat value
    val gen10 = gen1.setAndGetChild(false)
    val gen11 = gen1.setAndGetChild(true)

    tree
  }

  test("build small tree") {
    val tree = buildSmallTree
  }

  test("build medium tree") {
    val tree = buildMediumTree
  }

  test("predict samll tree w/ row") {}
  test("predict medium tree w/ row") {}

  test("predict samll tree w/ dataframe") {}
  test("predict medium tree w/ dataframe") {}
}
