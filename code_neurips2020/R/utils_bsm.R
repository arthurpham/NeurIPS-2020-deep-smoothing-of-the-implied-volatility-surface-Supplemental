gbsm_price <- function(S, K, tau, Ir, Id, sig, type = "C") {
  d1 <- (log(S / K) + Ir - Id + 0.5 * sig^2 * tau) / (sig * sqrt(tau))
  d2 <- d1 - sig * sqrt(tau)

  if (type == "C") {
    value <- exp(-Id) * S * pnorm(d1) - K * exp(-Ir) * pnorm(d2)
  } else if (type == "P") {
    value <- K * exp(-Ir) * pnorm(-d2) - exp(-Id) * S * pnorm(-d1)
  } else {
    stop("unknown type")
  }

  return(value)
}

gbsm_iv <- function(S, K, tau, Ir, Id, pi, type, style = "E") {
  ## Compute implied volatility given parameters, type, and market price Return NA if cannot find a vol satisfying 1bp < IV < 200%
  if (style == "E") {
    f <- function(x) {
      gbsm_price(S, K, tau, Ir, Id, x, type) - pi
    }
  } else if (style == "A") {
    require(fOptions)
    f <- function(x) {
      ty <- ifelse(type == "P", "pa", "ca")
      n <- max(150, round(500 * tau))
      pi - CRRBinomialTreeOption(
        TypeFlag = ty, S = S, X = K, Time = tau,
        r = Ir / tau, b = (Ir - Id) / tau, sigma = x, n = n
      )@price
    }
  } else {
    stop("not European nor American?")
  }

  iv <- NA
  try(
    {
      iv <- uniroot(
        f = f, lower = 1e-04, upper = 9.99, maxiter = 1e+05,
        tol = .Machine$double.eps / 2
      )$root
    },
    silent = T
  )

  return(iv)
}

gbsm_greek <- function(S, K, tau, Ir, Id, sig, type, name) {
  d1 <- (log(S / K) + Ir - Id + 0.5 * sig^2 * tau) / (sig * sqrt(tau))
  if (name == "delta") {
    if (type == "C") {
      return(exp(-Id) * pnorm(d1))
    }
    if (type == "P") {
      return(-exp(-Id) * pnorm(-d1))
    }
  } else if (name == "vega") {
    ## same for call and puts
    return(S * exp(-Id) * dnorm(d1) * sqrt(tau))
  } else {
    stop("Unknown greek")
  }
}

## some wrappers
bsm_price <- function(S, K, tau, r, d, sig, type = "C") gbsm_price(S = S, K = K, tau = tau, Ir = r * tau, Id = d * tau, sig = sig, type = type)
bsm_iv <- function(S, K, tau, r, d, pi, type) gbsm_iv(S = S, K = K, tau = tau, Ir = r * tau, I = d * tau, pi = pi, type = type)
bsm_greek <- function(S, K, tau, r, d, sig, type, name) gbsm_greek(S = S, K = K, tau = tau, Ir = r * tau, Id = d * tau, sig = sig, type = type, name = name)
