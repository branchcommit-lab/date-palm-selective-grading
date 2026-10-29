# Experimental records

The CSV files are recorded experimental outputs, not results from newly trained models. The C11 development evaluation includes 835 sources and 12,525 variants; the final C1/C2 evaluation includes 834 locked-test sources and 12,510 observations. These are distinct populations.

`metadata/source_split.csv` records 5,836 source images, of which 5,828 are eligible for the source-group-separated partitions: train 3,603, development 835, calibration 556 and locked test 834. Original photograph files and masks are external. Source image identifiers and cryptographic file digests permit alignment with the publisher's source collection.

`metadata/development_only_selection.json` documents development-based settings, while the Dandhi-targeted candidate and its selected probability blend are documented in the associated C2 records. Results mode does not refit parameters or grading models.
