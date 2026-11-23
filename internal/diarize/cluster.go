package diarize

import (
	"slices"
)

// cluster is agglomerative-cluster --threshold=t --read-costs=false
// (Kaldi AgglomerativeClusterer, single pass, min-clusters 1): average
// linkage on the costs -score, merging while the cheapest pair costs at
// most -t. It returns Kaldi's 1-based labels, numbered in order of the
// final cluster ids. All cost arithmetic is float32, like Kaldi's, so ties
// and stop decisions match.
//
// Kaldi switches to a two-pass scheme above 32767 points; this port always
// uses the single pass.
func cluster(scores []float32, n int, t float32) []int {
	threshold := -t
	// cost[sa*n+sb] holds the summed cost between the clusters stored in
	// slots sa and sb. A merged cluster keeps the slot of its first parent.
	// Kaldi reads only the upper triangle (i < j) of the score matrix.
	cost := make([]float32, n*n)
	for i := range n {
		for j := i + 1; j < n; j++ {
			cost[i*n+j], cost[j*n+i] = -scores[i*n+j], -scores[i*n+j]
		}
	}
	into := make([]int32, n)       // slot -> slot it was merged into
	slot := make([]int32, 2*n+1)   // cluster id -> slot
	size := make([]float32, 2*n+1) // cluster id -> number of points
	pos := make([]int32, 2*n+1)    // cluster id -> index in active, -1 if merged
	active := make([]int32, 0, n)
	var q pairHeap
	for i := range n {
		id := int32(i + 1)
		slot[id], size[id], pos[id], into[i] = int32(i), 1, int32(i), int32(i)
		active = append(active, id)
		for j := i + 1; j < n; j++ {
			if c := cost[i*n+j]; c <= threshold {
				q = append(q, pair{c, id, int32(j + 1)})
			}
		}
	}
	for i := range pos[n+1:] {
		pos[n+1+i] = -1
	}
	q.init()
	remove := func(id int32) {
		i := pos[id]
		last := active[len(active)-1]
		active[i], pos[last] = last, i
		active = active[:len(active)-1]
		pos[id] = -1
	}
	count := int32(n) //nolint:gosec // G115: the n*n cost matrix keeps n far below MaxInt32
	nextCompact := len(active) / 2
	for len(active) > 1 && len(q) > 0 {
		p := q.pop()
		if pos[p.a] < 0 || pos[p.b] < 0 {
			continue
		}
		// MergeClusters(a, b): the new cluster takes the next id and a's slot.
		count++
		sa, sb := slot[p.a], slot[p.b]
		slot[count], size[count], into[sb] = sa, size[p.a]+size[p.b], sa
		remove(p.a)
		remove(p.b)
		for _, k := range active {
			sk := slot[k]
			c := cost[int(sa)*n+int(sk)] + cost[int(sb)*n+int(sk)]
			cost[int(sa)*n+int(sk)], cost[int(sk)*n+int(sa)] = c, c
			norm := size[count] * size[k]
			if c/norm <= threshold {
				q.push(pair{c / norm, k, count}) // k < count
			}
		}
		pos[count] = int32(len(active)) //nolint:gosec // G115: len(active) <= n
		active = append(active, count)
		// Dropping entries of merged clusters does not change the result:
		// Kaldi pops and skips them.
		if len(active) <= nextCompact {
			q = slices.DeleteFunc(q, func(e pair) bool { return pos[e.a] < 0 || pos[e.b] < 0 })
			q.init()
			nextCompact = len(active) / 2
		}
	}
	// AssignClusters: labels follow the ascending order of cluster ids.
	slices.Sort(active)
	label := make([]int, n) // slot -> label
	for l, id := range active {
		label[slot[id]] = l + 1
	}
	labels := make([]int, n)
	for i := range n {
		s := int32(i)
		for into[s] != s {
			s = into[s]
		}
		labels[i] = label[s]
	}
	return labels
}

// pair is a queue entry: average cost and the two cluster ids, a < b.
type pair struct {
	cost float32
	a, b int32
}

// less is the order of Kaldi's std::priority_queue with std::greater on
// (cost, (a<<16)+b): cheapest first, ties by the smaller encoded key.
func (x pair) less(y pair) bool {
	if x.cost != y.cost {
		return x.cost < y.cost
	}
	if x.a != y.a {
		return x.a < y.a
	}
	return x.b < y.b
}

// pairHeap is a binary min-heap of pairs ordered by less.
type pairHeap []pair

// init establishes the heap order.
func (h pairHeap) init() {
	for i := len(h)/2 - 1; i >= 0; i-- {
		h.down(i)
	}
}

// push adds p to the heap.
func (h *pairHeap) push(p pair) {
	*h = append(*h, p)
	s := *h
	i := len(s) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !s[i].less(s[parent]) {
			break
		}
		s[i], s[parent] = s[parent], s[i]
		i = parent
	}
}

// pop removes and returns the cheapest pair.
func (h *pairHeap) pop() pair {
	s := *h
	top := s[0]
	s[0] = s[len(s)-1]
	*h = s[:len(s)-1]
	h.down(0)
	return top
}

// down moves element i down until the heap order holds.
func (h pairHeap) down(i int) {
	n := len(h)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		m := l
		if r := l + 1; r < n && h[r].less(h[l]) {
			m = r
		}
		if !h[m].less(h[i]) {
			return
		}
		h[i], h[m] = h[m], h[i]
		i = m
	}
}
