import streamlit as st


def model_params_ui(name, task, prefix):
    """Full manual hyperparameter controls, hidden behind one Advanced expander."""
    p = {}
    with st.expander(f"⚙️ {name}", expanded=name == "Neural Network"):
        if name == "Logistic Regression":
            p["C"] = st.number_input("C", .001, 100., 1., key=f"{prefix}_C")
            p["max_iter"] = st.number_input("Max iterations", 100, 10000, 2000, 100, key=f"{prefix}_mi")
            cw = st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_cw")
            p["class_weight"] = None if cw == "None" else cw
        elif name == "Naive Bayes":
            p["var_smoothing"] = st.number_input("Variance smoothing", 1e-12, 1e-5, 1e-9, format="%.1e", key=f"{prefix}_vs")
        elif name in {"Decision Tree", "Random Forest", "Extra Trees"}:
            limit = st.toggle("Limit max depth", value=name == "Decision Tree", key=f"{prefix}_limit")
            p["max_depth"] = st.slider("Max depth", 1, 50, 8, key=f"{prefix}_md") if limit else None
            p["min_samples_split"] = st.slider("Min samples split", 2, 30, 2, key=f"{prefix}_split")
            p["min_samples_leaf"] = st.slider("Min samples leaf", 1, 20, 1, key=f"{prefix}_leaf")
            if name in {"Random Forest", "Extra Trees"}:
                p["n_estimators"] = st.slider("Trees", 50, 1200, 300, 50, key=f"{prefix}_trees")
            if task == "classification":
                cw = st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_cw2")
                p["class_weight"] = None if cw == "None" else cw
        elif name in {"Gradient Boosting", "AdaBoost"}:
            p["n_estimators"] = st.slider("Estimators", 50, 1000, 200, 50, key=f"{prefix}_n")
            p["learning_rate"] = st.number_input("Learning rate", .001, 1., .05, key=f"{prefix}_lr")
            if name == "Gradient Boosting":
                p["max_depth"] = st.slider("Tree depth", 1, 8, 3, key=f"{prefix}_depth")
        elif name == "Hist Gradient Boosting":
            p["max_iter"] = st.slider("Boosting iterations", 50, 1000, 250, 50, key=f"{prefix}_iter")
            p["learning_rate"] = st.number_input("Learning rate", .001, 1., .08, key=f"{prefix}_hlr")
            p["max_leaf_nodes"] = st.slider("Max leaf nodes", 5, 127, 31, key=f"{prefix}_leafnodes")
            p["l2_regularization"] = st.number_input("L2 regularization", 0., 100., 0., key=f"{prefix}_l2")
        elif name == "KNN":
            p["n_neighbors"] = st.slider("K neighbors", 1, 50, 7, key=f"{prefix}_k")
            p["weights"] = st.selectbox("Weights", ["uniform", "distance"], key=f"{prefix}_w")
            p["p"] = st.selectbox("Distance", [1, 2], format_func=lambda x: "Manhattan" if x == 1 else "Euclidean", key=f"{prefix}_p")
        elif name == "SVM":
            p["C"] = st.number_input("C", .01, 100., 1., key=f"{prefix}_svmc")
            p["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"], key=f"{prefix}_kernel")
            p["gamma"] = st.selectbox("Gamma", ["scale", "auto"], key=f"{prefix}_gamma")
            if task == "classification":
                cw = st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_svmcw")
                p["class_weight"] = None if cw == "None" else cw
            else:
                p["epsilon"] = st.number_input("Epsilon", .001, 10., .1, key=f"{prefix}_eps")
        elif name == "XGBoost":
            p["n_estimators"] = st.slider("Boosting rounds", 50, 1500, 300, 50, key=f"{prefix}_xn")
            p["max_depth"] = st.slider("Max depth", 1, 15, 6, key=f"{prefix}_xd")
            p["learning_rate"] = st.number_input("Learning rate", .001, 1., .05, key=f"{prefix}_xlr")
            p["subsample"] = st.slider("Row subsample", .4, 1., .9, .05, key=f"{prefix}_xs")
            p["colsample_bytree"] = st.slider("Column sample", .4, 1., .9, .05, key=f"{prefix}_xc")
            c1, c2 = st.columns(2)
            with c1:
                p["reg_alpha"] = st.number_input("L1 alpha", 0., 100., 0., key=f"{prefix}_xa")
            with c2:
                p["reg_lambda"] = st.number_input("L2 lambda", 0., 100., 1., key=f"{prefix}_xlam")
        elif name == "LightGBM":
            p["n_estimators"] = st.slider("Trees", 50, 1500, 300, 50, key=f"{prefix}_lgbn")
            p["learning_rate"] = st.number_input("Learning rate", .001, 1., .05, key=f"{prefix}_lgblr")
            p["num_leaves"] = st.slider("Num leaves", 7, 255, 31, key=f"{prefix}_lgbleaves")
            p["max_depth"] = st.slider("Max depth (-1 = unlimited)", -1, 30, -1, key=f"{prefix}_lgbdepth")
            p["subsample"] = st.slider("Row subsample", .4, 1., .9, .05, key=f"{prefix}_lgbsub")
            p["colsample_bytree"] = st.slider("Column sample", .4, 1., .9, .05, key=f"{prefix}_lgbcol")
            if task == "classification":
                cw = st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_lgbcw")
                p["class_weight"] = None if cw == "None" else cw
        elif name == "Polynomial Regression":
            p["degree"] = st.slider("Polynomial degree", 2, 3, 2, key=f"{prefix}_polydegree")
        elif name == "Neural Network":
            mode = st.segmented_control("Architecture", ["Auto", "Custom"], default="Auto", key=f"{prefix}_mode")
            if mode == "Auto":
                p["_auto_architecture"] = True
                st.info("Auto uses encoded input width and training-set size to choose a conservative network.")
            else:
                p["_auto_architecture"] = False
                n = st.slider("Hidden layers", 1, 8, 2, key=f"{prefix}_layers")
                sizes = []
                cols = st.columns(min(4, n))
                for i in range(n):
                    with cols[i % len(cols)]:
                        sizes.append(int(st.number_input(f"Layer {i + 1} neurons", 1, 2048, max(8, 64 // (2 ** i)), key=f"{prefix}_u{i}")))
                p["hidden_layer_sizes"] = tuple(sizes)
                st.code("Input → " + " → ".join(map(str, sizes)) + " → Output")
            c1, c2 = st.columns(2)
            with c1:
                p["activation"] = st.selectbox("Activation", ["relu", "tanh", "logistic", "identity"], key=f"{prefix}_act")
                p["solver"] = st.selectbox("Solver", ["adam", "sgd", "lbfgs"], key=f"{prefix}_solver")
                p["alpha"] = st.number_input("L2 alpha", 0., 10., .0001, format="%.6f", key=f"{prefix}_alpha")
            with c2:
                p["max_iter"] = st.slider("Max epochs / iterations", 100, 5000, 800, 100, key=f"{prefix}_epochs")
                if p["solver"] != "lbfgs":
                    p["learning_rate_init"] = st.number_input("Initial learning rate", .00001, 1., .001, format="%.5f", key=f"{prefix}_nnlr")
                    p["learning_rate"] = st.selectbox("LR schedule", ["constant", "adaptive", "invscaling"], key=f"{prefix}_schedule")
                    custom = st.toggle("Custom batch size", False, key=f"{prefix}_cb")
                    p["batch_size"] = st.slider("Batch size", 8, 1024, 64, 8, key=f"{prefix}_bs") if custom else "auto"
            if p["solver"] != "lbfgs":
                p["early_stopping"] = st.toggle("Early stopping", True, key=f"{prefix}_es")
                if p["early_stopping"]:
                    p["validation_fraction"] = st.slider("Validation fraction", .05, .30, .10, .01, key=f"{prefix}_vf")
                    p["n_iter_no_change"] = st.slider("Patience", 5, 100, 15, key=f"{prefix}_pat")
            else:
                p["early_stopping"] = False
        elif name in {"Ridge", "Lasso", "ElasticNet"}:
            p["alpha"] = st.number_input("Alpha", .00001, 1000., 1. if name == "Ridge" else .1, format="%.5f", key=f"{prefix}_alpha_reg")
            if name == "ElasticNet":
                p["l1_ratio"] = st.slider("L1 ratio", 0., 1., .5, .05, key=f"{prefix}_l1r")
            if name in {"Lasso", "ElasticNet"}:
                p["max_iter"] = 5000
    return p
