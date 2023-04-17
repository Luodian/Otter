def generate_matches(args):
    """
    Genreate the matches:
    Args:
        match_str: a pre-defined string to determine the matches. 
            A typical match_str is like first:attention_mse,hidden_mse
    Return:
        matches: a list of dict 
    """
    match_str = args.intermediate_matches
    T_config = args.model_T_config
    S_config = args.model_S_config
    T_layers = T_config['num_hidden_layers']
    S_layers = S_config['num_hidden_layers']
    T_d, S_d = T_config['hidden_size'], S_config['hidden_size']

    matches = []

    match_strs = match_str.split(",")
    # determin the matched-layer
    # NOTE: the number of layer begin with 1
    for match_str in match_strs:
        match_layer_method, match_method = match_str.split(":")
        match_layers = []
        if match_layer_method.startswith('first'):
            if match_layer_method == 'first':
                match_layers = [[i, i] for i in range(S_layers)]
            else:
                # The match_layer_method could be first10, wil use the first 10 layers of teacher model
                first_k = int(match_layer_method.replace('first', ''))
                match_layers = [[i, i] for i in range(first_k)]
        elif match_layer_method.startswith('last'):
            if match_layer_method == 'last':
                match_layers = [[i + (T_layers - S_layers), i]
                                for i in range(S_layers)]
            else:
                # The match_layer_method could be first10, wil use the first 10 layers of teacher model
                last_k = int(match_layer_method.replace('last', ''))
                match_layers = [[i + T_layers - last_k, S_layers - 1 - i]
                                for i in range(last_k)]

        elif match_layer_method == 'gap':
            factor = T_layers // S_layers
            match_layers = []
            for i in range(S_layers):
                match_layers.append([(i + 1) * factor - 1, i])

        # match_method = match_method.split(",")
        if T_d != S_d:
            proj = ["linear", S_d, T_d]
        else:
            proj = None
        # for method in match_method:
        method = match_method

        base_match = {
            "loss": method,
            "weight": 1,
        }
        if method.startswith("attention"):
            base_match["feature"] = "attention"
        elif method.endswith("relation"):
            base_match["feature"] = "kvs"
        elif method in ['hidden_mse', "mmd", "gram", "cos", "pkd"]:
            base_match["feature"] = "hidden"
            if proj is not None:
                base_match["proj"] = proj

        # Assign each layer match
        for layer_T, layer_S in match_layers:
            match = base_match.copy()
            if method.startswith("attention") or method.endswith("relation"):
                match.update({"layer_T": layer_T, "layer_S": layer_S})
            elif method in ['hidden_mse', 'cos', 'pkd']:
                match.update({
                    "layer_T": layer_T + 1,
                    "layer_S": layer_S + 1,
                })
            elif method in ['mmd', 'gram']:
                match.update({
                    "layer_T": [layer_T + 1, layer_T + 1],
                    "layer_S": [layer_S + 1, layer_S + 1]
                })

            else:
                raise NotImplementedError
            matches.append(match)

        if method in ['hidden_mse']:
            match = base_match.copy()
            match.update({"layer_T": 0, "layer_S": 0})
            matches.append(match)
        elif method in ["mmd", "gram"]:
            match = base_match.copy()
            match.update({"layer_T": [0, 0], "layer_S": [0, 0]})
            matches.append(match)

    print(f"Matches: {matches}")
    return matches

def generate_matches_ofa(args):
    """
    Genreate the matches:
    Args:
        match_str: a pre-defined string to determine the matches.
            A typical match_str is like first:attention_mse,hidden_mse
    Return:
        matches: a list of dict
    """
    match_str = args.intermediate_matches
    T_config = args.model_T_config
    S_config = args.model_S_config


    T_d, S_d = T_config['d_model'], S_config['d_model']

    matches = []

    match_strs = match_str.split(",")
    # determin the matched-layer
    # NOTE: the number of layer begin with 1
    for match_str in match_strs:
        match_layer_method, match_method, xcoder = match_str.split(":")
        T_layers = T_config['%s_layers' % xcoder]
        S_layers = S_config['%s_layers' % xcoder]
        match_layers = []
        if match_layer_method.startswith('first'):
            if match_layer_method == 'first':
                match_layers = [[i, i] for i in range(S_layers)]
            else:
                # The match_layer_method could be first10, wil use the first 10 layers of teacher model
                first_k = int(match_layer_method.replace('first', ''))
                match_layers = [[i, i] for i in range(first_k)]
        elif match_layer_method.startswith('last'):
            if match_layer_method == 'last':
                match_layers = [[i + (T_layers - S_layers), i]
                                for i in range(S_layers)]
            else:
                # The match_layer_method could be first10, wil use the first 10 layers of teacher model
                last_k = int(match_layer_method.replace('last', ''))
                match_layers = [[i + T_layers - last_k, S_layers - 1 - i]
                                for i in range(last_k)]

        elif match_layer_method == 'gap':
            factor = T_layers // S_layers
            match_layers = []
            for i in range(S_layers):
                match_layers.append([(i + 1) * factor - 1, i])

        # match_method = match_method.split(",")
        if T_d != S_d:
            proj = ["linear", S_d, T_d]
        else:
            proj = None
        # for method in match_method:
        method = match_method

        base_match = {
            "loss": method,
            "weight": 1,
            "xcoder": xcoder
        }
        if method.startswith("attention"):
            base_match["feature"] = "%s_attention" % xcoder
        elif method.endswith("relation"):
            base_match["feature"] = "%s_kvs" % xcoder
        elif method in ['hidden_mse', "mmd", "gram", "cos", "pkd"]:
            base_match["feature"] = "%s_hidden" % xcoder
            if proj is not None:
                base_match["proj"] = proj

        # Assign each layer match
        for layer_T, layer_S in match_layers:
            match = base_match.copy()
            if method.startswith("attention") or method.endswith("relation"):
                match.update({"layer_T" : layer_T, "layer_S" : layer_S})
            elif method in ['hidden_mse', 'cos', 'pkd']:
                match.update({
                    "layer_T": layer_T + 1,
                    "layer_S": layer_S + 1,
                })
            elif method in ['mmd', 'gram']:
                match.update({
                    "layer_T": [layer_T + 1, layer_T + 1],
                    "layer_S": [layer_S + 1, layer_S + 1]
                })

            else:
                raise NotImplementedError
            matches.append(match)

        if method in ['hidden_mse']:
            match = base_match.copy()
            match.update({"layer_T": 0, "layer_S": 0})
            matches.append(match)
        elif method in ["mmd", "gram"]:
            match = base_match.copy()
            match.update({"layer_T": [0, 0], "layer_S": [0, 0]})
            matches.append(match)

    print(f"Matches: {matches}")
    return matches


L3_attention_mse = [{
    "layer_T": 4,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 8,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}]

L3_attention_ce = [{
    "layer_T": 4,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 8,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}]

L3_attention_mse_sum = [{
    "layer_T": 4,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 8,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}]

L3_attention_ce_mean = [{
    "layer_T": 4,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 8,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}]

L3_hidden_smmd = [{
    "layer_T": [0, 0],
    "layer_S": [0, 0],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [4, 4],
    "layer_S": [1, 1],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [8, 8],
    "layer_S": [2, 2],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [12, 12],
    "layer_S": [3, 3],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}]

L3n_hidden_mse = [{
    "layer_T": 0,
    "layer_S": 0,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 384, 768]
}, {
    "layer_T": 4,
    "layer_S": 1,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 384, 768]
}, {
    "layer_T": 8,
    "layer_S": 2,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 384, 768]
}, {
    "layer_T": 12,
    "layer_S": 3,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 384, 768]
}]

L3_hidden_mse = [{
    "layer_T": 0,
    "layer_S": 0,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 4,
    "layer_S": 1,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 8,
    "layer_S": 2,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 3,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}]

# ######################L4################
L4_attention_mse = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_mse",
    "weight": 1
}]

L4_attention_ce = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_ce",
    "weight": 1
}]

L4_attention_mse_sum = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_mse_sum",
    "weight": 1
}]

L4_attention_ce_mean = [{
    "layer_T": 3,
    "layer_S": 1,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "attention",
    "loss": "attention_ce_mean",
    "weight": 1
}]

L4_hidden_smmd = [{
    "layer_T": [0, 0],
    "layer_S": [0, 0],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [3, 3],
    "layer_S": [1, 1],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [6, 6],
    "layer_S": [2, 2],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [9, 9],
    "layer_S": [3, 3],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [12, 12],
    "layer_S": [4, 4],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}]

L4t_hidden_sgram = [{
    "layer_T": [0, 0],
    "layer_S": [0, 0],
    "feature": "hidden",
    "loss": "gram",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": [3, 3],
    "layer_S": [1, 1],
    "feature": "hidden",
    "loss": "gram",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": [6, 6],
    "layer_S": [2, 2],
    "feature": "hidden",
    "loss": "gram",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": [9, 9],
    "layer_S": [3, 3],
    "feature": "hidden",
    "loss": "gram",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": [12, 12],
    "layer_S": [4, 4],
    "feature": "hidden",
    "loss": "gram",
    "weight": 1,
    "proj": ["linear", 312, 768]
}]

L4t_hidden_mse = [{
    "layer_T": 0,
    "layer_S": 0,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 3,
    "layer_S": 1,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 6,
    "layer_S": 2,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 9,
    "layer_S": 3,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}, {
    "layer_T": 12,
    "layer_S": 4,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 312, 768]
}]

# ##########L6#############
L6_hidden_smmd = [{
    "layer_T": [0, 0],
    "layer_S": [0, 0],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [2, 2],
    "layer_S": [1, 1],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [4, 4],
    "layer_S": [2, 2],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [6, 6],
    "layer_S": [3, 3],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [8, 8],
    "layer_S": [4, 4],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [10, 10],
    "layer_S": [5, 5],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [12, 12],
    "layer_S": [6, 6],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}]

L6_hidden_mse = [{
    "layer_T": 0,
    "layer_S": 0,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 2,
    "layer_S": 1,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 4,
    "layer_S": 2,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 6,
    "layer_S": 3,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 8,
    "layer_S": 4,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 10,
    "layer_S": 5,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}, {
    "layer_T": 12,
    "layer_S": 6,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1
}]

# electra-small
small_hidden_smmd = [{
    "layer_T": [0, 0],
    "layer_S": [0, 0],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [2, 2],
    "layer_S": [2, 2],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [4, 4],
    "layer_S": [4, 4],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [6, 6],
    "layer_S": [6, 6],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [8, 8],
    "layer_S": [8, 8],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [10, 10],
    "layer_S": [10, 10],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}, {
    "layer_T": [12, 12],
    "layer_S": [12, 12],
    "feature": "hidden",
    "loss": "mmd",
    "weight": 1
}]

small_hidden_mse = [{
    "layer_T": 0,
    "layer_S": 0,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}, {
    "layer_T": 2,
    "layer_S": 2,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}, {
    "layer_T": 4,
    "layer_S": 4,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}, {
    "layer_T": 6,
    "layer_S": 6,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}, {
    "layer_T": 8,
    "layer_S": 8,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}, {
    "layer_T": 10,
    "layer_S": 10,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}, {
    "layer_T": 12,
    "layer_S": 12,
    "feature": "hidden",
    "loss": "hidden_mse",
    "weight": 1,
    "proj": ["linear", 256, 768]
}]

matches = {
    'L3_attention_mse': L3_attention_mse,
    'L3_attention_mse_sum': L3_attention_mse_sum,
    'L3_attention_ce': L3_attention_ce,
    'L3_attention_ce_mean': L3_attention_ce_mean,
    'L3n_hidden_mse': L3n_hidden_mse,
    'L3_hidden_smmd': L3_hidden_smmd,
    'L3_hidden_mse': L3_hidden_mse,
    'L4_attention_mse': L4_attention_mse,
    'L4_attention_mse_sum': L4_attention_mse_sum,
    'L4_attention_ce': L4_attention_ce,
    'L4_attention_ce_mean': L4_attention_ce_mean,
    'L4t_hidden_mse': L4t_hidden_mse,
    'L4_hidden_smmd': L4_hidden_smmd,
    'L4t_hidden_sgram': L4t_hidden_sgram,
    'L6_hidden_mse': L6_hidden_mse,
    'L6_hidden_smmd': L6_hidden_smmd,
    'small_hidden_mse': small_hidden_mse,
    'small_hidden_smmd': small_hidden_smmd
}
