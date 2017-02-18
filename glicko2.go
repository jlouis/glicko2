// Package glicko2 implements the Glicko 2 ranking system. It defines
// a method which can compute the ranking of a player, given plays
// against opponents for a tournament.
//
// The code follows the Glicko2 paper quite precisely, except that it uses Ridder's method
// for finding roots. It does a bit of allocation, but in my tests it is by far the fastest code among
// solutions in Erlang, Go and OCaml.
// There are numerous places of low-hanging fruit, should the code prove to run too slowly.
package glicko2

import (
	"math"
)

// The standard constants used in Glicko 2
const (
	scaling = 173.7178 // Scaling factor for Glicko2
	ε       = 0.000001 // When to stop seeking a more precise result
)

// The Opponent interface is used to define opponents in a tournament. It is given as a slice to the
// ranker and every element in the slice is treated as a game that was played in the tournament.
// The interface must return the rating via R(); the rating deviance via RD() and the current σ value via
// Sigma(). The value SJ() encodes the outcome of the match. If the player for which we rank won the
// match, return 1.0. If the player lost, return 0.0. If the match was a draw, return 0.5. Note that Glicko2
// is not really strong and handling draws.
type Opponent interface {
	R() float64
	RD() float64
	Sigma() float64
	SJ() float64
}

// Internal opponent structure used to store some commonly used calculations
// TODO: Get rid of the memory allocations and simply compute the values when they
// are needed. It is not nearly worth it to build up garbage for something which could
// be calculated on the fly when needed instead.
type opp struct {
	muj   float64
	phij  float64
	gphij float64
	emmp  float64
	sj    float64
}

// scale scales the range down to the one used by Glicko2.
// projects values as seen by the outside world unto the internally used
// range. This is essentially for converting between the Glicko 1 and 2 system,
// by allowing you to reuse values from Glicko 1. Also note that this makes it
// possible to compare values to the ELO system.
func scale(r float64, rd float64) (mu float64, phi float64) {
	mu = (r - 1500.0) / scaling
	phi = (rd / scaling)
	return mu, phi
}

// unscale is the inverse of the scale function.
// brings back values to the normal ELO/Glicko 1 scaling
func unscale(mup float64, phip float64) (float64, float64) {
	rp := scaling*mup + 1500.0
	rdp := scaling * phip
	return rp, rdp
}

// g computes a commonly used internal value
func g(phi float64) float64 {
	return (1 / math.Sqrt(1+3*phi*phi/(math.Pi*math.Pi)))
}

// e computes a commonly used internal value
func e(mu float64, muj float64, phij float64) float64 {
	return (1 / (1 + math.Exp(-g(phij)*(mu-muj))))
}

// scaleOpponents processes the opponents and returns a new slice with the internal opponent structure
func scaleOpponents(mu float64, os []Opponent) (res []opp) {
	res = make([]opp, len(os))
	for i, o := range os {
		muj, phij := scale(o.R(), o.RD())
		res[i] = opp{muj, phij, g(phij), e(mu, muj, phij), o.SJ()}
	}

	return res
}

// updateRating computes the new updated rating based on the opponents
func updateRating(sopp []opp) float64 {
	s := 0.0
	for _, o := range sopp {
		s += o.gphij * o.gphij * o.emmp * (1 - o.emmp)
	}

	return 1 / s
}

// computeDelta is part of the deviation and volatility update
func computeDelta(v float64, sopp []opp) float64 {
	s := 0.0
	for _, o := range sopp {
		s += o.gphij * (o.sj - o.emmp)
	}

	return v * s
}

// volK is part of the volatility update
func volK(f func(float64) float64, a float64, tau float64) float64 {
	k := 0.0
	c := a - k*math.Sqrt(tau*tau)
	i := 0
	for ; f(c) < 0.0; k += 1.0 {
		c = a - k*math.Sqrt(tau*tau)
		i++
		if i > 10000 {
			panic("volK exceeded")
		}
	}

	return c
}

// sign returns the sign of a float64 value
// returns 1.0 for positive, -1.0 for negative and 0 for 0.
func sign(x float64) float64 {
	if x < 0 {
		return -1.0
	} else if x > 0 {
		return 1.0
	} else {
		return 0.0
	}
}

// computeVolatility computes the next volatility for the player
// The algorithm used here deviates from the Glicko2 paper. In the code below, we construct a function, f,
// and the goal is to find a root of this function. Originally, the paper used a Newton-Rhapson method, but
// this often leads to numerical instability and infinite cycling. The Glicko2 paper suggests Illinois method
// as a solution. But we have implemented Ridder's method here instead. It often completes in fewer iterations
// and the author has seen no inifinite cycles.
//
// If an infinite cycle happens, it is interesting and the code will panic.
func computeVolatility(sigma float64, phi float64, v float64, delta float64, tau float64) float64 {
	a := math.Log(sigma * sigma)
	phi2 := phi * phi
	f := func(x float64) float64 {
		ex := math.Exp(x)
		d2 := delta * delta
		a2 := phi2 + v + ex
		p2 := (x - a) / (tau * tau)
		p1 := (ex * (d2 - phi2 - v - ex)) / (2 * a2 * a2)
		return (p1 - p2)
	}

	var b float64
	if delta*delta > phi*phi+v {
		b = math.Log(delta*delta - phi*phi - v)
	} else {
		b = volK(f, a, tau)
	}

	fa := f(a)
	fb := f(b)

	var c, fc, d, fd float64
	for i := 100; i > 0; i-- {
		if math.Abs(b-a) <= ε {
			return math.Exp(a / 2)
		}

		c = (a + b) * 0.5
		fc = f(c)
		d = c + (c-a)*(sign(fa-fb)*fc)/math.Sqrt(fc*fc-fa*fb)
		fd = f(d)

		if sign(fd) != sign(fc) {
			a = c
			b = d
			fa = fc
			fb = fd
		} else if sign(fd) != sign(fa) {
			b = d
			fb = fd
		} else {
			a = d
			fa = fd
		}

	}

	panic("Exceeded iterations")
}

// phiStar returns another component of the glicko ranking
func phiStar(sigmap float64, phi float64) float64 {
	return math.Sqrt(phi*phi + sigmap*sigmap)
}

// newRating returns the new rating of the player
func newRating(phis float64, mu float64, v float64, sopp []opp) (float64, float64) {
	phip := 1 / math.Sqrt(
		(1/(phis*phis))+(1/v))
	s := 0.0
	for _, o := range sopp {
		s += o.gphij * (o.sj - o.emmp)
	}
	mup := mu + (phip*phip)*s

	return mup, phip
}

// Rank ranks a player given a list of opponents.
// In order to rank a player, you must supply the rating r, the rating deviance rd and the volatility, sigma (σ). You must also supply a list
// of opponents as a []Opponent. And you must supply the configuration parameter tau (τ)
// Good values of tau are between 0.3 and 1.2. You will have to tune your data set to find a good tau by running a prediction algorithm.
// The function returns three values, nr, nrd, nsigma for the new rating, rating deviation and sigma/volatility value respectively.
func Rank(r, rd, sigma float64, opponents []Opponent, tau float64) (nr, nrd, nsigma float64) {
	mu, phi := scale(r, rd)
	sopps := scaleOpponents(mu, opponents)
	v := updateRating(sopps)
	delta := computeDelta(v, sopps)

	nsigma = computeVolatility(sigma, phi, v, delta, tau)
	phistar := phiStar(nsigma, phi)
	mup, phip := newRating(phistar, mu, v, sopps)
	nr, nrd = unscale(mup, phip)

	return nr, nrd, nsigma
}

// Skip is used when a player skips a tournament.
// In the case where a player skips a tournament, we will keep the rating, r, and the volatility, sigma, of the player the same, but the rating deviation, rd, will
// change. This function returns the new rating deviation of the player
func Skip(r, rd, sigma float64) float64 {
	mu, phi := scale(r, rd)
	phi = phiStar(sigma, phi)
	_, phi = unscale(mu, phi)
	return phi
}
