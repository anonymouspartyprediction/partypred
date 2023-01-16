Core pipeline: construct_graphs -> gcn.py/gat.py/label_prop_baseline.py. These take path to configuration file as argument, e.g. construct_graphs.py config/main_experiment.json.

If using gcn, can then use second_stage.py to combine embeddings from different input types and get final prediction.

Run_pipeline.py works with single_model.py and single_model_job.py to run the whole pipeline, particularly if using a slurm cluster to run many splits in parallel.

To run particular experiments:

* Figure 2: modularity_vs_accuracy.py

* Table 5: config/main_experiment.json (config for pipeline with GCN), config/test_projection.json (with label propagation), config/test_projection_GAT (with GAT)

* Table 8, Accuracy: config/no_allrelations_filter.json (with GCN), config/no_allrelations_unprojected (with label propagation)

* Table 8, Coverage: calculate_applicability.py

* Table 8, Speed: users_retrievable_per_15m.py

* Table 9: config/politicians_unsupervised.json (with GCN), config/politicians_unprojected.json (with label propagation; label_prop_w_politicians.py for this one)

* Table 11: config/main_experiment.json (semisupervised GCN, with parse_semisupervised_output.py, or with second_stage.py to add RF), config/test_semisupervised.json (unsupervised GCN), config/test_GAT_unprojected (GAT, with parse_semisupervised_output.py, or with second_stage_GAT.py to add RF)

* Table 12: config/main_experiment.json (Pro. GCN-1L and Pro. Label Prop.), config/test_projection.json (Dir. GCN-1L and Dir. Label Prop), config/test_projection_twolayer.json (Dir. GCN-2L), config/test_GAT.json (Pro. GAT), config/test_GAT_unprojected.json (Dir. GAT)


All above need paths to data set in config/common_data_paths.json.
We do not make original data publicly available in order to ensure ethical usage. It is available upon request for academic replication purposes only (in dehydrated form to comply with Twitter API terms of use).






