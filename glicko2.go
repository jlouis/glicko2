// Package glicko2 implements the Glicko 2 ranking system. It defines
// a method which can compute the ranking of a player, given plays
// against opponents for a tournament.
package glicko2

import (
	"math"
)

// The standard constants used in Glicko 2
const (
	scaling = 173.7178 // Scaling factor for Glicko2
	ε       = 0.000001 // When to stop seeking a more precise result
)

type PlayerName struct {
	Id	string // Players Id
	M	string // The map the player is playing on
}

type Player struct {
	Id        PlayerName  // Player Identification
	Name      string  // Player name
	R         float64 // Player ranking
	Rd        float64 // Ranking deviation
	Sigma     float64 // Volatility
	Active	bool // Is the player currently active?
}

type Opponent struct {
	Idx	int	// Player index
	Sj float64 // Match score
}

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


func scaleOpponents(mu float64, os []Opponent, players []Player) (res []opp) {
	res = make([]opp, len(os))
	for i, o := range os {
		muj, phij := scale(players[o.Idx].R, players[o.Idx].Rd)
		res[i] = opp{muj, phij, g(phij), e(mu, muj, phij), o.Sj}
	}

	return res
}

func updateRating(sopp []opp) float64 {
	s := 0.0
	for _, o := range sopp {
		s += o.gphij * o.gphij * o.emmp * (1 - o.emmp)
	}

	return 1 / s
}

func computeDelta(v float64, sopp []opp) float64 {
	s := 0.0
	for _, o := range sopp {
		s += o.gphij * (o.sj - o.emmp)
	}

	return v * s
}

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

func sign(x float64) float64 {
	if x < 0 {
		return -1.0
	} else if x > 0 {
		return 1.0
	} else {
		return 0.0
	}
}

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
	if delta*delta > phi*phi + v {
		b = math.Log(delta*delta - phi*phi - v)
	} else {
		b = volK(f, a, tau)
	}

	fa := f(a)
	fb := f(b)

	var c, fc, d, fd float64
	for i := 100; i > 0 ; i-- {
		if math.Abs(b-a) <= ε {
			return math.Exp(a / 2)
		} else {
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
	}

	panic("Exceeded iterations")
}

func phiStar(sigmap float64, phi float64) float64 {
	return math.Sqrt(phi*phi + sigmap*sigmap)
}

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


func Rank(r, rd, sigma float64, opponents []Opponent, players []Player, tau float64) (nr, nrd, nsigma float64) {
	mu, phi := scale(r, rd)
	sopps := scaleOpponents(mu, opponents, players)
	v := updateRating(sopps)
	delta := computeDelta(v, sopps)

	nsigma = computeVolatility(sigma, phi, v, delta, tau)
	phistar := phiStar(nsigma, phi)
	mup, phip := newRating(phistar, mu, v, sopps)
	nr, nrd = unscale(mup, phip)

	return nr, nrd, nsigma
}
