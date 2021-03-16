from pathlib import Path

def create_save_path(args) -> Path:
    """Generates saving path given the script arguments."""
    if args.dir:
        dir_path = Path(args.dir)
    else:
        dir_path = Path(__file__).parents[1] / Path('saved_models') 

    if args.name:
        filename = args.name
    else:
        filename = f"walker_D{args.duration}_N{args.n_gens}_STD{args.std:.2E}.pth"
    
    dir_path.mkdir(exist_ok=True)
    return Path(dir_path) / Path(filename)