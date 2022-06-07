local rho   = 2
local Cv    = 3
local sigma = 4
local T0    = 1e5
local gamma = 5/3

local c     = 2.99792458e+8
local a     = 7.56576651e-16
local tau   = 2.27761040e+9
local omega = 2.13503497e+1

local rad = function(x,y,z)
  return sqrt(x^2 + y^2)
end

local boundary_flux = function(x,y,z,t)
  local dr_dx = cos(atan(y/x))
  local dr_dy = sin(atan(y/x))
  local exponent = exp(-tau * t)
  local Trad     = T0 * (1 + 0.5 * exponent)
  local F        = c / (3 * sigma) *
                 a * Trad^4 * 0.5 * exponent * omega * sin(omega * rad(x,y,z))
  return F*dr_dx, F*dr_dy, 0
end

initial_conditions = {
  time_step =  0.1/tau/((2^3)^ref_levels)/2,
  velocity = 0,
  radiation_energy = function(x,y,z,t)
    return a * (T0 * 1.5)^4 * (1 + 0.5 * cos(omega * rad(x,y,z)))
  end,
  material = {
    [1] = {
      density = rho,
      energy = function(x,y,z,t)
        return Cv * T0 * (1 - 0.5 * cos(omega * rad(x,y,z)))
      end,
      volume_fraction = 1,
    },
  },
}

boundary_conditions = {
  reflecting = {1,3},
  specified = {2,4},
  specifications = {
    [2] = {
      radiation_flux = boundary_flux
    },
    [4] = {
      radiation_flux = boundary_flux
    }
  }
}

sources = {
  radiation_energy = function(x,y,z,t)
    local r = rad(x,y,z)
    local exponent = exp(-tau * t)
    local cosine   = cos(omega * r)
    local Tmat     = T0 * (1 - 0.5 * exponent * cosine)
    local Trad     = T0 * (1 + 0.5 * exponent)
    local E        = a * Trad^4 * (1 + 0.5 * exponent * cosine)
    return -0.5 * tau * exponent * a * Trad^3 *
           (4 * T0 + (Trad + 2 * T0 * exponent) * cosine)
           +
           c * exponent * a * Trad^4 / (6 * sigma) *
           (omega^2 * cosine + omega * sin(omega * r) / r)
           -
           c * sigma * (a * Tmat^4 - E)
  end,
  material = {
    [1] = {
      energy_per_volume = function(x,y,z,t)
        local exponent = exp(-tau * t)
        local cosine   = cos(omega * rad(x,y,z))
        local Tmat     = T0 * (1 - 0.5 * exponent * cosine)
        local Trad     = T0 * (1 + 0.5 * exponent)
        local E        = a * Trad^4 * (1 + 0.5 * exponent * cosine)
        return rho * Cv * 0.5 * T0 * tau * exponent * cosine
               +
               c * sigma * (a * Tmat^4 - E)
      end
    }
  }
}

solution = {
  radiation_energy =
    function(x, y, z, t)
      local r = sqrt(x^2 + y^2)
      local exponent = exp(-tau * t)
      local Trad     = T0 * (1 + 0.5 * exponent)
      return a * Trad^4 * (1 + 0.5 * exponent * cos(omega * r))
    end,
  thermal_energy =
    function(x, y, z, t)
      local r = sqrt(x^2 + y^2)
      return Cv * T0 * (1 - 0.5 * exp(-tau * t) * cos(omega * r))
    end,
  material_temperature =
    function(x, y, z, t)
      local r = sqrt(x^2 + y^2)
      return T0 * (1 - 0.5 * exp(-tau * t) * cos(omega * r))
    end
}
