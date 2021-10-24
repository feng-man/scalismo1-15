package c8

import java.awt.Color

import scalismo.geometry._
import scalismo.common._
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh._
import scalismo.io.{StatisticalModelIO, MeshIO}
import scalismo.statisticalmodel._
import scalismo.numerics.UniformMeshSampler3D
import scalismo.kernels._

import scalismo.ui.api._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.{DenseMatrix, DenseVector}
object c8 {

  def main(args: Array[String]) : Unit = {

    // setting a seed for the random generator to allow for reproducible results
    implicit val rng = scalismo.utils.Random(42)

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    /** begin */
    /** begin */
    /** 加载和预处理数据集： */
    val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new java.io.File("datasets/bfm.h5")).get

    val modelGroup = ui.createGroup("modelGroup")
    val ssmView = ui.show(modelGroup, model, "model")

    /** Fitting observed data using Gaussian process regression */
    val idNoseTip = PointId(8156)
    val noseTipReference = model.reference.pointSet.point(idNoseTip)
    val noseTipMean = model.mean.pointSet.point(idNoseTip)
    val noseTipDeformation = (noseTipReference - noseTipMean) * 2.0

    val noseTipDomain = UnstructuredPointsDomain3D(IndexedSeq(noseTipReference))
    val noseTipDeformationField = DiscreteField3D(noseTipDomain,  IndexedSeq(noseTipDeformation))

    val observationGroup = ui.createGroup("observation")
    ui.show(observationGroup, noseTipDeformationField, "noseTip")

    val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))

    val regressionData = IndexedSeq((noseTipReference, noseTipDeformation, noise))

    val gp : LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = model.gp.interpolate(TriangleMeshInterpolator3D())
    val posteriorGP : LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = LowRankGaussianProcess.regression(gp, regressionData)

    gp.posterior(regressionData)

    val posteriorSample: DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] =
      posteriorGP.sampleAtPoints(model.reference)
    val posteriorSampleGroup = ui.createGroup("posteriorSamples")
    for (i <- 0 until 10) {
      ui.show(posteriorSampleGroup, posteriorSample, "posteriorSample")
    }

    /** Posterior of a StatisticalMeshModel: */
    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.01)
    val pointOnLargeNose = noseTipReference + noseTipDeformation
    val discreteTrainingData = IndexedSeq((PointId(8156), pointOnLargeNose, littleNoise))
    val meshModelPosterior: PointDistributionModel[_3D, TriangleMesh] = model.posterior(discreteTrainingData)

    val posteriorModelGroup = ui.createGroup("posteriorModel")
    ui.show(posteriorModelGroup, meshModelPosterior, "NoseyModel")

    val largeNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 5.0)
    val discreteTrainingDataLargeNoise = IndexedSeq((PointId(8156), pointOnLargeNose, largeNoise))
    val discretePosteriorLargeNoise = model.posterior(discreteTrainingDataLargeNoise)
    val posteriorGroupLargeNoise = ui.createGroup("posteriorLargeNoise")
    ui.show(posteriorGroupLargeNoise, discretePosteriorLargeNoise, "NoisyNoseyModel")



    /** end  */
    /** end  */
  }
}