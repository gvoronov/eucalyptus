package eucalyptus.test

import org.scalatest.FunSuite
import org.scalatest.Assertions._

import koalas.datasets.bikesharing.BikeSharing
import koalas.row.Row
import koalas.datavalue._
import koalas.numericalops.NumericalOps._

import eucalyptus.tree._

class TreeSuite extends FunSuite {
  val df = BikeSharing.loadDayDF

  def buildSmallTree: BiTree = {
    val tree = new BiTree(new BiNode("atemp", 0.5))

    val df0 = df.filter(_[NumericalValue]("atemp") < 0.5)
    val df1 = df.filter(_[NumericalValue]("atemp") >= 0.5)

    val gen0: Node = tree.root.setAndGetChild(
      false, new RegressionLeaf(df0.select[NumericalValue]("cnt")))
    val gen1: Node = tree.root.setAndGetChild(
      true, new BiNode("holiday", 0.5,
      Map(
        CategoricalValue("0", "holiday") -> NumericalValue(0),
        CategoricalValue("1", "holiday") -> NumericalValue(1)
      ))
    )

    val df10 = df1.filter(_[ClassCategoricalValue]("holiday") == CategoricalValue("0", "holiday"))
    val df11 = df1.filter(_[ClassCategoricalValue]("holiday") == CategoricalValue("1", "holiday"))

    val gen10 = gen1.setAndGetChild(false, new RegressionLeaf(df10.select[NumericalValue]("cnt")))
    val gen11 = gen1.setAndGetChild(true, new RegressionLeaf(df11.select[NumericalValue]("cnt")))

    tree
  }

  // def buildMediumTree: BiTree = {
  //   val tree = new BiTree(BiNode("cnt", 500))
  //
  //   val gen0 = tree.root.setAndGetChild(false)
  //   val gen1 = tree.root.setAndGetChild(true)
  //
  //   val gen00 = gen1.setAndGetChild(false)
  //   val gen01 = gen1.setAndGetChild(true)
  //   // Look at a simple cat value
  //   val gen10 = gen1.setAndGetChild(false)
  //   val gen11 = gen1.setAndGetChild(true)
  //
  //   tree
  // }

  test("build small tree") {
    val tree = buildSmallTree

    val gen0 = tree.root.getChild(false)
    val gen1 = tree.root.getChild(true)

    val gen10 = gen1.getChild(false)
    val gen11 = gen1.getChild(true)

    assertThrows[RuntimeException] {gen0.getChild(true)}
    assertThrows[RuntimeException] {gen10.getChild(true)}
    assertThrows[RuntimeException] {gen11.getChild(true)}

    assert(gen0.predict[NumericalValue](null) ~= 3433.045092838196)
    assert(gen10.predict[NumericalValue](null) ~= 5647.289017341041)
    assert(gen11.predict[NumericalValue](null) ~= 5557.375)
  }

  test("predict samll tree w/ row") {
    val tree = buildSmallTree

    assert(tree.predict[NumericalValue](df.irow(0)) ~= 3433.04509283819)
    assert(tree.predict[NumericalValue](df.irow(500)) ~= 5647.289017341041)
    assert(tree.predict[NumericalValue](df.irow(149)) ~= 5557.375)
  }

  test("predict samll tree w/ dataframe") {
    val tree = buildSmallTree
    val prediction = tree.predict[NumericalValue](df)
    val actual = df[NumericalValue]("cnt")
    val mse = ((prediction :- actual):**2).mean

    assert(mse ~= 2525308.704797109)
  }

  // test("build medium tree") {
  //   val tree = buildMediumTree
  // }
  // test("predict medium tree w/ row") {}
  // test("predict medium tree w/ dataframe") {}
}
