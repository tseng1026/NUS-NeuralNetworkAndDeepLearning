def ComputeLength():
	total_len = 0
	total_seg = 0

	fp = open("training_segment.txt", "r")
	for k in range(1460):
		# read the segment
		content = fp.readline()
		segment = [int(frame) for frame in content.split()]

		# crop the segment (train)
		for frame in range(len(segment) - 1):
			length = segment[frame+1] - segment[frame]
			total_len += length
			total_seg += 1

	fp = open("test_segment.txt", "r")
	for k in range(252):
		# read the segment
		content = fp.readline()
		segment = [int(frame) for frame in content.split()]

		# crop the segment (test)
		for frame in range(len(segment) - 1):
			length = segment[frame+1] - segment[frame]
			total_len += length
			total_seg += 1

	return total_len // total_seg
