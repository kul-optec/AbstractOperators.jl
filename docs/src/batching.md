# Batch operators

The batch operator is particularly useful in scenarios where the same operation needs to be applied across multiple dimensions of data simultaneously. For instance, in machine learning, it can be used to process batches of input data through a neural network, ensuring that each input in the batch undergoes the same transformations. Similarly, in image processing, the batch operator can apply filters or transformations to a set of images in parallel, significantly improving efficiency. Technically it is the optimized special case of `DCAT`, in which the diagonally concatenated operators are all the same. This operator is essential for tasks that require uniform processing across large datasets, enabling scalable and consistent operations.

An even more advanced version is the spreading batch operator, which allows applying different operators over one or more "spreading" batch dimensions, while the rest of the batch dimensions behaves as a "normal" batch dimension. For example, in medical imaging (e.g. CT and MRI) it is common to acquire 3D-images of the patient, and these volumetric aquisitions are repeated over a specified time. This results in a 4-dimensional data set for which, let's say, we want to apply different image processing operators over the 2D slices of the volumetric "time frames", but we want to do the same procedure along the time dimension.

```@docs
BatchOp
ThreadingStrategy
```
