from liftoff import parse_opts

from src.fractal_renderer.make_fractaldb import generate_dset


def run(opt):
    print(opt)
    opt.csv_name = f"{opt.run_id:05d}.csv"
    generate_dset(opt)


if __name__ == "__main__":
    run(parse_opts())
