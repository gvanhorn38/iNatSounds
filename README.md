# iNatSounds
iNatSounds Dataset

## Details

| Super Category | Species Count | Train Recordings | Train Mini Recordings | Val Recordings | Test Recordings |
| ---- | ---- | ---- | ---- | ---- | ---- |
Birds|1,486|414,847|74,300|14,860|x|
Insects|2,526|663,682|126,300|25,260|x|
Amphibians|170|46,252|8,500|1,700|x|
Mammals|246|68,917|12,300|2,460|x|
Reptiles|313|86,830|15,650|3,130|x|
||||||
Total|10,000|2,686,843|500,000|100,000|500,000|


## Evaluation


## Guidelines

## Annotation Format

```
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation],
  "licenses" : [license]
}

info{
  "year" : int,
  "version" : str,
  "description" : str,
  "contributor" : str,
  "url" : str,
  "date_created" : str,
}

image{
  "id" : int,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "license" : int,
  "rights_holder" : str,
  "date": str,
  "latitude": float,
  "longitude": float,
  "location_uncertainty": int,
}

category{
  "id" : int,
  "name" : str,
  "common_name" : str,
  "supercategory" : str,
  "kingdom" : str,
  "phylum" : str,
  "class" : str,
  "order" : str,
  "family" : str,
  "genus" : str,
  "specific_epithet" : str,
  "image_dir_name" : str,
}

annotation{
  "id" : int,
  "image_id" : int,
  "category_id" : int
}

license{
  "id" : int,
  "name" : str,
  "url" : str
}
```

### Annotation Format Notes:

## Terms of Use
By downloading this dataset you agree to the following terms:

1. You will abide by the [iNaturalist Terms of Service](https://www.inaturalist.org/pages/terms).
2. You will use the data only for non-commercial research and educational purposes.
3. You will NOT distribute the dataset recordings.
4. UMass Amherst makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
5. You accept full responsibility for your use of the data and shall defend and indemnify UMass Amherst, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

## Data

The dataset files are available through the AWS Open Data Program:
  * [Train Images [224GB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz)
      * s3://ml-inat-competition-datasets/2021/train.tar.gz
      * Running `md5sum train.tar.gz` should produce `e0526d53c7f7b2e3167b2b43bb2690ed`
      * Images have a max dimension of 500px and have been converted to JPEG format
      * Untaring the images creates a directory structure like `train/category/image.jpg`. This may take a while.
  * [Train Annotations [221MB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz)
      * s3://ml-inat-competition-datasets/2021/train.json.tar.gz
      * Running `md5sum train.json.tar.gz` should produce `38a7bb733f7a09214d44293460ec0021`
  * [Train Mini Images [42GB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz)
      * s3://ml-inat-competition-datasets/2021/train_mini.tar.gz
      * Running `md5sum train_mini.tar.gz` should produce `db6ed8330e634445efc8fec83ae81442`
      * Images have a max dimension of 500px and have been converted to JPEG format
      * Untaring the images creates a directory structure like `train_mini/category/image.jpg`. This may take a while.
  * [Train Mini Annotations [45MB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz)
      * s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz
      * Running `md5sum train_mini.json.tar.gz` should produce `395a35be3651d86dc3b0d365b8ea5f92`
  * [Validation Images [8.4GB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz)
      * s3://ml-inat-competition-datasets/2021/val.tar.gz
      * Running `md5sum val.tar.gz` should produce `f6f6e0e242e3d4c9569ba56400938afc`
      * Images have a max dimension of 500px and have been converted to JPEG format
      * Untaring the images creates a directory structure like `val/category/image.jpg`. This may take a while.
  * [Validation Annotations [9.4MB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz)
      * s3://ml-inat-competition-datasets/2021/val.json.tar.gz
      * Running `md5sum val.json.tar.gz` should produce `4d761e0f6a86cc63e8f7afc91f6a8f0b`
  * [Test Images [43GB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz)
      * s3://ml-inat-competition-datasets/2021/public_test.tar.gz
      * Running `md5sum public_test.tar.gz` should produce `7124b949fe79bfa7f7019a15ef3dbd06`
      * Images have a max dimension of 500px and have been converted to JPEG format
      * Untaring the images creates a directory structure like `public_test/image.jpg`.
  * [Test Info [21MB]](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz)
      * s3://ml-inat-competition-datasets/2021/public_test.json.tar.gz
      * Running `md5sum public_test.json.tar.gz` should produce `7a9413db55c6fa452824469cc7dd9d3d`
