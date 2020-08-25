def to_process(clip, center):
    import ipdb; ipdb.set_trace()

    # Lengths and frames
    clip_len = len(clip)
    first_frame = clip[:,:,:1]
    last_frame  = clip[:,:,-1:]
    l_pad = max(0, 8-center) #leftpad
    r_pad = max(0, (8+center)-clip_len) #rightpad

    # Prepare chunks to create subclip
    first_chunk  = torch.cat([first_frame]*l_pad, dim=2)
    middle_chunk = clip[:,:,(8-center):(8+center)]
    last_cunk    = torch.cat([last_frame]*r_pad, dim=2)
    return torch.cat([first_chunk, middle_chunk, last_cunk], dim=2)