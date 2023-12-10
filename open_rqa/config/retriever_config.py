PROD_SEARCH_CONFIG = {
	'mmr': {
		'name': 'mmr',
		'kwargs': {
			'lambda_mult': 0.75,
		}
	},
	'custom': {
		'name': 'similarity',  # langchain only accepts certain names
		'kwargs': {
			'method': 'cosine_w_bm25',
			'semantic_threshold': 0.7,
			'normalize_L2': True,
			'semantic_k': 20,
			'k': 4  # this is the final k
		}
	},
	'v1.0': {
		'name': 'similarity',
		'kwargs': {
			'method': 'cosine',
			'normalize_L2': True,
			'score_threshold': 0.2,
			'final_k': 4,
		}
	}
}


SEARCH_CONFIG = {
	'cosine': {
		"search_type": 'similarity',
		"search_kwargs": {
			'method': 'cosine',
			'normalize_L2': True,
			'score_threshold': 0.5,
			'k': 4  # this is the final k
		},
	},
	'inner_product': {
		"search_type": 'similarity',
		"search_kwargs": {
			'method': 'inner_product',  # so normalize_L2 = False
			'normalize_L2': False,
			'k': 4  # this is the final k
		},
	},
	'cosine_w_bm25': {
		'search_type': 'similarity',
		'search_kwargs': {
			'method': 'cosine_w_bm25',
			'normalize_L2': True,
			'semantic_threshold': 0.7,
			'semantic_k': 20,
			'k': 4  # this is the final k
		}
	},
	'inner_product_w_bm25': {
		"search_type": 'similarity',
		"search_kwargs": {
			'method': 'inner_product_w_bm25',  # so normalize_L2 = False
			'normalize_L2': False,
			'semantic_k': 4,  # 4//2=2 comes from inner_product, the rest from bm25
			'k': 4  # this is the final k
		},
	}
}