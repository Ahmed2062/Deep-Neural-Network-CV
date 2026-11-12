"""Console formatting helpers."""

WIDTH = 64


def header(title):
    print("\n" + "=" * WIDTH)
    print(title.center(WIDTH))
    print("=" * WIDTH)


def table(rows, columns):
    """Print a list of dicts as an aligned table. columns: [(key, title, fmt)]."""
    cells = [[fmt.format(r[key]) for key, _, fmt in columns] for r in rows]
    titles = [t for _, t, _ in columns]
    widths = [max(len(t), *(len(c[i]) for c in cells)) for i, t in enumerate(titles)]
    line = "  ".join(t.rjust(w) for t, w in zip(titles, widths))
    print(line)
    print("-" * len(line))
    for c in cells:
        print("  ".join(v.rjust(w) for v, w in zip(c, widths)))
