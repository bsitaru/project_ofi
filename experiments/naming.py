from ml_collections import ConfigDict


def args_to_name(args):
    r = [args.experiment.name]
    r += ["issize", str(args.experiment.in_sample_size)]
    r += ["ossize", str(args.experiment.os_size)]
    r += ["rolling", str(args.experiment.rolling)]
    r += ["seltype", args.selector.type]
    r += ["lvl", str(args.selector.levels)]
    if args.selector.volume_normalize:
        r += ["volnrm"]
    if args.processor.normalize:
        r += ["indvnrm"]
    if 'pca' in args.processor:
        r += ["pca", str(args.processor.pca)]
    if 'multipca' in args.processor:
        r += ["multipca", str(args.processor.multipca.groups), str(args.processor.multipca.components)]
    if 'clustering' in args:
        r += ["clustering", str(args.clustering.n_clusters)]
    if 'neighbours' in args:
        r += ["neighbours", str(args.neighbours.neigh_size)]
    r += ["regtype", str(args.regression.type)]
    return "_".join(r)
