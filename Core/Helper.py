def discretize(value, screen_width, num_bins):
    bin_size = screen_width // num_bins
    index = int(value // bin_size)
    if value % bin_size > 0.5 * bin_size:
        index += 1
    return index
