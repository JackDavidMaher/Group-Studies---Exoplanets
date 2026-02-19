# Licensed under a 3-clause BSD style license - see LICENSE.rst

import bisect
import warnings
import copy

import numpy as np
import numpy.ma as ma
from .custom_exceptions import EngineInputError, DataConfigurationError
from .constants import MIN_CLIP
from .exposure_factory import ExposureFactory
from .utils import spectrum_resample
class Detector:

    def __init__(self, config, webapp=False):
        """
        All functions that are detector-specific will live under children of this class

        DetectorNoise will use the this class (or rather, its detector-specific subclasses) to produce noise.
        """
        self.webapp = webapp
        self.detector_config = config["detector_config"]
        self.subarray_config = config["subarray_config"]
        self.readout_pattern_config = config["readout_pattern_config"]
        self.input_detector = config["input_detector"]

        # slopemode is given among the detector parameters that otherwise really define exposure
        self.slopemode = self.input_detector.get('slopemode', 'multiaccum')

        self.__dict__.update(self.detector_config)
        self.exposure_spec = self.get_exposure_pars(config)

    def get_exposure_pars(self, name="pattern_name"):
        """
        Define exposure parameters from instrument and environment.

        Parameters
        ----------
        name: str
            Optional name to assign ExposureSpecification instance

        Returns
        -------
        exposure_spec: pandeia.engine.exposure.ExposureSpecification instance
            Exposure parameters encapsulated within ExposureSpecification instance
        """
        # These are values that get set when the Instrument is instantiated, either from defaults
        # or passed configuration data.
        # These are defined in the reference data as "ramp_config" and apply to a whole instrument

        exp_conf = {"input_detector": self.input_detector, "subarray": self.subarray_config,
            "readout_pattern": self.readout_pattern_config, "detector_config": self.detector_config}

        exposure_spec = ExposureFactory(self.det_type, config=exp_conf, webapp=self.webapp)

        return exposure_spec

    def set_saturation(self, flag):
        # this function turns saturation on and off - if the flag is True, we 
        # compute saturation. If toggle is False, we do not.
        if not flag:
            self.fullwell = np.inf

    def set_readnoise(self, flag):
        # this function turns readnoise on and off - if the flag is False,
        # we set the readnoise to 0.
        if not flag:
            self.rn = 0

    def set_excess(self, flag):
        # this function turns the various excess noise components on and off - if the flag is False,
        # we set var_fudge, excessp1, and excessp2 to 0.
        if not flag:
            self.var_fudge = 1
            self.excessp1 = 0
            self.excessp2 = 0

    def get_slope_variance(self, rate, unsat):
        # This function allows different slope variance functions (used for MULTIACCUM TA)
        slope_var, slope_rn_var = self.slope_variance(rate, unsat)
        return slope_var, slope_rn_var

    def scale_exposure_time(self, extracted):
        raise NotImplementedError("Reverse ETC calculations for {} are not implemented.".format(type(self)))

    def set_time(self, new_time):
        raise NotImplementedError("Reverse ETC calculations for {} are not implemented.".format(type(self)))

    def to_resels_scalar(self, wave, values, refwave, squared=False):
        """
        Extract scalar value from array in resels.
        Requires an array, a resel (read from the detector), and the reference wavelength
        Uses astropy 1D box model to apply the resel value.
        The one advance this has over previous Pandeia behavior is that you DO get the (value) at the
        average position of your reference wavelength.

        This works in any case where you're dropping the wavelength dimension, so also cube->2D

        Parameters
        ----------
        wave: ndarray
            the wavelength array, in microns
        values: ndarray
            the value array 
        refwave: float
            the reference wavelength in microns
        squared: bool
            True in case the value array is a noise array that must be summed in quadrature (JETC-3020)

        Returns
        -------
        value: float
            the value in resels
        """
        values = ma.fix_invalid(values, fill_value=0.0).data
        if len(wave) > 1:
            if hasattr(self,"pixels_per_resel"):
                # find out what fractional index position the reference wavelength is at
                cenidx = np.interp(refwave, wave, np.arange(len(wave)))
                mask_pix = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
                mask = np.array(
                    [
                        0,
                        cenidx - (self.pixels_per_resel + 1) / 2.0,
                        cenidx - (self.pixels_per_resel - 1) / 2.0,
                        cenidx + (self.pixels_per_resel - 1) / 2.0,
                        cenidx + (self.pixels_per_resel + 1) / 2.0,
                        len(wave)-1
                    ]
                )
                boxfilter = np.interp(np.arange(len(wave)), mask, mask_pix)
                if len(values.shape) == 3: # if we're doing a cube, the boxfilter needs to be reshaped
                    boxfilter = boxfilter.reshape(-1,1,1)

                # sum the array, linearly, or in quadrature. This is used for noise, and
                # is effectively the same as multiplying the values by the square root of
                # the boxfilter, just as the noise extraction is over the square of the noise aperture.
                if squared:
                    resel_value = np.sum(values ** 2 * boxfilter) / \
                                  (self.pixels_per_resel ** 2)  # the 1D values are simply extracted over the pixel-area of one resel
                    resel_value = np.sqrt(resel_value)
                else:
                    resel_value = np.sum(
                        values * boxfilter) / self.pixels_per_resel  # the 1D values are simply extracted over the pixel-area of one resel

            else:
                # there are no resels but it is a 1D array
                wave_index = (np.abs(wave - refwave)).argmin()
                resel_value = values[wave_index]
        else:
            # it is a single-element array
            resel_value = values[0]

        return resel_value

    # TODO: Resels for downsampling a cube

    def to_resels_array(self, wave, values):
        """
        Convert array to resels.
        Requires an array and a resel (read from the instrument)

        Parameters
        ----------
        wave: ndarray
            the wavelength array, in microns
        values: ndarray
            the value array 

        Returns
        -------
        rebinwave: ndarray
            the rebinned wavelength array, in microns
        array: ndarray
            the rebinned array
        """
        if hasattr(self,"pixels_per_resel"):
            if self.pixels_per_resel != 1 and len(wave) > 1:
                # spectrum_resample cannot handle NaNs, so we need to save the nanmask and set nans to 0.
                nanmask = np.isfinite(values) 
                values = ma.fix_invalid(values, fill_value=0.0).data

                rebinwave = wave[::self.pixels_per_resel] # this is the same operation conducted by PyETC
                resel_array = spectrum_resample(values, wave, rebinwave)

                # rebin the nanmask (all elements in resel must be true= not NaN for the resel to not be NaN) and reapply
                rebinmask = [np.all(nanmask[x:x+self.pixels_per_resel]) for x in range(0, len(values), self.pixels_per_resel)]
                return rebinwave, resel_array*rebinmask
        # if none of the above were true, return the originals
        return wave, values

class Detector_MultiAccum(Detector):

    def get_unsaturated(self, slope, full_saturation=2):
        """
        Calculate the number of unsaturated groups in each pixel, given a specified slope and full well value.
        There is a minimum sensible value of unsaturated number of groups that defines full saturation.
        The default is 2, but this can potentially be set to a higher value, or even 1(!) for certain observing modes.
        It is not expected that a value of 0 can ever be anything but saturated.

        Parameters
        ----------
        slope: ndarray
            The measured slope of the ramp (ie, the rate) per pixel

        full_saturation: positive integer
            The minimum number of groups allowed to define an unsaturated measurement.

        Returns
        -------
        unsat_ngroups: MaskedArray
            The number of unsaturated groups present in the ramp. The mask separates pixels that have full saturation
            from those that do not. The former will not have a valid noise measurement and cannot be used in a strategy.

        """

        # get the number of groups to saturation
        ngroups_to_sat = self.get_before_sat(slope)

        # the number of unsaturated groups is clipped by the ngroups_to_sat value for every pixel
        max_ngroups = ngroups_to_sat.clip(0,self.exposure_spec.ngroup)

        # Mask values that are fully saturated - if the maximum before saturation is less than full_saturation, it's
        # clearly saturated.
        unsat_ngroups = ma.masked_less(max_ngroups, full_saturation)

        return unsat_ngroups

    def get_saturation_fraction(self, slope):
        """
        This method returns a map of the fraction of saturation that each pixel reaches, given the current
        exposure setup.

        Parameters
        ----------
        slope: ndarray
            The measured slope of the ramp (ie, the rate) per pixel

        """
        # we clip to 1e-10 to avoid division by zero errors even if the slope is tiny.
        return self.exposure_spec.saturation_time / (self.fullwell / slope.clip(MIN_CLIP, np.max([MIN_CLIP, np.max(slope)])))

    def _get_variance_fudge(self, wave):
        """
        Some instruments (e.g. MIRI) have measured noise properties that are not well explained
        or modeled by Pandeia's current noise model. This method returns the multiplicative fudge factor
        that is to be applied to the pixel rate variance to match IDT expectations. The fudge factor can be
        chromatic so needs to accept an array of wavelengths to interpolate over. By default this returns
        a 1's array the same length as wave scaled by the 'var_fudge' scalar value.

        Parameters
        ----------
        wave: 1D numpy.ndarray
            Wavelengths to interpolate chromatic variance over
        """
        if hasattr(self, 'var_fudge'):
            var_fudge = self.var_fudge
        else:
            var_fudge = 1
        
        return var_fudge * np.ones(len(wave))

    def get_before_sat(self, slope):
        """
        This method returns a map of the maximum number of groups each pixel can be exposed to, before saturation.
        The formula is calculated by equating the time-to-saturation to the saturation time formula and isolating
        ngroup. The resulting fractional value is then rounded down to an integer.

        These equations are derived by setting saturation_time = time_to_saturation in equation 7 (for NIR H2RG
        detectors) or equation 6 (for MIR SiAs detectors, now out of the timing document revision, JWST-STScI-006013-A)
        of the timing document, Valenti et al. 2017, JWST-STScI-006013, and solving for n_groups.

        Technically speaking, we are getting the groups BEFORE saturation by rounding the number of groups down to the
        nearest integer. If groups_before_sat is EXACTLY an integer (which is unlikely), we will get one partially
        saturated pixel.

        Parameters
        ----------
        slope: ndarray
            The measured slope of the ramp (ie, the rate) per pixel

        fullwell: positive integer
            The number of electrons defining a full well (beyond which the pixel reads become unusable for science
            due to saturation).
        """
        # we clip to 1e-10 to avoid division by zero errors even if the slope is tiny.
        time_to_saturation = self.fullwell / slope.clip(MIN_CLIP, np.max([MIN_CLIP, np.max(slope)]))

        groups_before_sat = self._get_groups_before_sat(time_to_saturation)

        # The groups_before_sat value should never be less than 0 - that would imply saturation before we started
        # collecting any photons.
        groups_before_sat = np.maximum(groups_before_sat,np.zeros_like(groups_before_sat))

        return np.floor(groups_before_sat)

    def slope_variance(self, rate, unsat_ngroups):
        """
        Calculate the variance of a specified MULTIACCUM slope.

        Inputs
        ------
        rate: ndarray
            The measured slope of the ramp (ie, the rate) per pixel
            This can be a one-dimensional wavelength spectrum (for a 1D ETC)
            or a three-dimensional cube ([wave,x,y] for a 3D ETC).

        unsat_ngroups: ndarray
            Number of unsaturated groups for each pixel

        Returns
        -------
        slope_var: ndarray
            Variance associated with the input slope.
        slope_rn_var: ndarray
            The associated variance of the readnoise only
        """

        # Rename variables for ease of comparison with Robberto (35).
        # The noise calculation depends on the pixel rate BEFORE IPC convolution. fp_pix_variance takes that pre-IPC
        # rate and scales it by the quantum yield and Fano factor to get the per-pixel variance in the electron rate.
        variance_per_pix = rate['fp_pix_variance']

        # we discard any saturated groups. Copy unsat_ngroups, because we may modify it for rejected groups before using
        # it for noise calculations.
        n = ma.copy(unsat_ngroups)

        # removing with any pre- and post-rejected groups here. This is not stictly directly related to saturation, bu
        # behaves in a similar way (if there is <2 groups available, we cannot define a slope). The very bright regime
        # in which groups are not rejected can be implemented (in parts) by setting nprerej=npostrej=0.
        # Therefore we disable it here in that regime.
        if ((self.exposure_spec.nprerej != 0) or (self.exposure_spec.npostrej != 0)):
            # If a group is already rejected because of saturation, there is no need to reject another at the end of
            # the ramp.
            unsat_pixels = (self.exposure_spec.ngroup-np.ceil(n))<self.exposure_spec.npostrej
            n[unsat_pixels] -= self.exposure_spec.nprerej
            # we do have to reject any post-rejected frames from all pixels
            n -= self.exposure_spec.npostrej

        # H2RG/H4RG and SiAs detectors need different sample values. (Issue #2996)
        m, tframe = self._get_sample()

        tgroup = self.exposure_spec.tgroup

        # Compute the variance of a MULTIACCUM slope using Robberto's formula (35). The rn_variance is also
        # calculated using the unsaturated number of groups.
        slope_rn_var = self.rn_variance(unsat_ngroups=n)
        # The final slope variance slope may also be worse than the theoretical best value.
        # (also from Glasse et al. 2015, PASP 127 686).
        # the variance product contains the dark current, so we do not need to add it here.
        slope_var = (6. / 5.) * (n ** 2. + 1.) / (n * (n ** 2. - 1.)) * \
            (variance_per_pix / self.exposure_spec.tgroup) * \
            (1. - (5. / 3.) * (m ** 2. - 1.) / (m * (n ** 2. + 1.)) * 
            (tframe / self.exposure_spec.tgroup)) + slope_rn_var

        # The default fill value for masked arrays is a finite number, so convert to ndarrays, and fill with NaNs
        # to make sure missing values are interpreted as truly undefined downstream.

        slope_var = ma.filled(slope_var, fill_value=np.nan)
        slope_rn_var = ma.filled(slope_rn_var, fill_value=np.nan)

        return np.asarray(slope_var, dtype=np.float32), np.asarray(slope_rn_var, dtype=np.float32)

    def slope_variance_ta(self, rate, unsat_ngroups):
        """
        Calculate the slope variance for TA modes

        Inputs
        ------
        rate: ndarray
            Rate

        unsat_ngroups: ndarray
           The number of unsaturated groups.

           If unsat_ngroups not supplied, the approximation is that the read noise is
           negligible for pixels with signal rates high enough to saturate in part of the ramp.
           This is probably always a good approximation.

        Returns
        -------
        var_rn: ndarray if unsat_ngroups is supplied, float if unsat_ngroups is not supplied.
           Variance associated with the read noise only.

        """

        # Rename variables for ease of comparison with Robberto (35).
        # The noise calculation depends on the pixel rate BEFORE IPC convolution. fp_pix_variance takes that pre-IPC
        # rate and scales it by the quantum yield and Fano factor to get the per-pixel variance in the electron rate.
        variance_per_pix = rate['fp_pix_variance']

        rn = self.rn
        n = unsat_ngroups  # we discard any saturated groups

        # H2RG/H4RG and SiAs detectors need different sample values. (Issue #2996)
        m, tframe = self._get_sample()

        nextract = self.exposure_spec.ngroup_extract

        # Throw error if the number of extract groups is >= number of groups.
        if self.exposure_spec.ngroup_extract > self.exposure_spec.ngroup:
            raise EngineInputError("Number of groups to extract, {}, must be less than or equal to the number of groups {}".format(
                self.exposure_spec.ngroup_extract, self.exposure_spec.ngroup))

        tgroup = self.exposure_spec.tgroup

        # The effective group time. This is the time from the first group to the next extracted group. For instance,
        # if ngroups=11 and nextract=3, textract = tgroup * 5
        textract = tgroup * (n-1)/(nextract-1)

        # The TA script will take the minimum of co-subtracted pairs. This has non-Gaussian error properties,
        # but a Monte Carlo Simulation indicates an improvement of 2/3 on the variance for the minimum of two
        # correlated pairs in a single ramp (as compared to a factor 1/2 for the average). It would be good to
        # demonstrate this analytically, but that has not yet been accomplished.

        # Number of possible pairs.
        npairs = nextract-1
        if npairs == 1:
            min_factor = 1.0
        elif npairs == 2:
            min_factor = 2./3.
        else:
            raise EngineInputError("Target acquisition currently only supports 2 or 3 extracted groups.")

        # Modification of MULTIACCUM formula for n=2 and a revised "effective" group time that takes into
        # account that only some groups obtained may be used to create the TA images.
        slope_rn_var = min_factor * (2*rn**2)/m/textract**2
        slope_var = min_factor * (variance_per_pix/textract) * (1.-(1./3.)*(m**2-1)/m * tframe/textract) + slope_rn_var

        # The default fill value for masked arrays is a finite number, so convert to ndarrays, and fill with NaNs
        # to make sure missing values are interpreted as truly undefined downstream.

        slope_var = ma.filled(slope_var, fill_value=np.nan)
        slope_rn_var = ma.filled(slope_rn_var, fill_value=np.nan)

        return np.asarray(slope_var, dtype=np.float32), np.asarray(slope_rn_var, dtype=np.float32)

    def rn_variance(self, unsat_ngroups=None):
        """
        Calculate the variance due to read noise only.

        Inputs
        ------
        rn: float
          Readnoise per pixel.

        unsat_ngroups: ndarray
           The number of unsaturated groups.

           If unsat_ngroups not supplied, the approximation is that the read noise is
           negligible for pixels with signal rates high enough to saturate in part of the ramp.
           This is probably always a good approximation.

        Returns
        -------
        var_rn: ndarray if unsat_ngroups is supplied, float if unsat_ngroups is not supplied.
           Variance associated with the read noise only.

        """
        if unsat_ngroups is None:
            n = self.exposure_spec.ngroup - (self.exposure_spec.nprerej + self.exposure_spec.npostrej)
        else:
            n = unsat_ngroups

        rn = self.rn

        # H2RG/H4RG and SiAs detectors need different sample values. (Issue #2996)
        m, tframe = self._get_sample()

        tgroup = self.exposure_spec.tgroup

        var_rn = 12. * rn ** 2. / (m * n * (n ** 2. - 1.) * tgroup ** 2.)

        # The readnoise on the slope may be worse than the theoretical best value
        # (see Glasse et al. 2015, PASP 127 686).
        if self.rn_fudge != 1:
            var_rn *= self.rn_fudge

        # Include the empirical correction for excess variance for long ramps
        # (Issue #2091)
        if self.excessp1 != 0.0 or self.excessp2 != 0.0:
            excess_variance = (12.0 * (n-1)/(n+1) * self.excessp1**2 - self.excessp2/np.sqrt(m)) / ((1 - n) * tgroup)**2
            var_rn += excess_variance
        return var_rn

    def _get_saturation_mask(self, rate):
        """
        Compute a numpy array indicating pixels with full saturation (2), partial saturation (1) and no saturation (0).

        Parameters
        ----------
        rate: None or 2D np.ndarray
            Detector plane rate image used to build saturation map from

        Returns
        -------
        mask: 2D np.ndarray
            Saturation mask image
        """
        saturation_mask = np.zeros(rate.shape)
        unsat_ngroups = self.get_unsaturated(rate, self.fullwell)

        saturation_mask[(unsat_ngroups < self.exposure_spec.ngroup)] = 1
        saturation_mask[(unsat_ngroups < 2)] = 2

        return saturation_mask

    def calc_cr_loss(self, ngroups, pix_cr_rate):
        """
        Calculate the effective loss of exposure time due to cosmic rays. This uses the cosmic ray (CR) event rate
        that is contained within the telescope configuration to calculate the odds of a cosmic rays
        hitting a pixel. Pixels that are hit by cosmic rays are assumed to have ramps that are valid before the
        CR, but not after. The exposure specification is used to adjust the expectation value of the exposure
        time (i.e. ngroups) for a CR-truncated ramp.

        See discussion and links in:

        - https://confluence.stsci.edu/pages/viewpage.action?spaceKey=JWST&title=2014-11-10+Effect+of+CRs+on+SNR
        - Robberto (2010) Technical Report JWST-STScI-001928

        for more information on implementation and the numbers used.

        Parameters
        ----------
        ngroups: int
            Number of groups over which to calculate CR losses.  This will usually be number of unsaturated groups.

        Returns
        -------
        cr_ngroups: float
            This is the input ngroups scaled by the mean loss of time due to cosmic ray events.
        """

        # this is the average fraction of the ramp that is lost upon a CR event
        if ngroups < self.mingroups:
            # if for some reason (e.g. using single read noise model) ngroup is less than the minimum needed
            # to get a good ramp fit, then the whole ramp is lost upon a CR event
            ramp_frac = 1.0
        else:
            # the more groups you have per ramp, the less of the ramp you lose to CRs on average.
            # this formalism takes into account that there's always a fraction that's totally lost.
            # in the limit of infinite reads this converges on half the ramp since CRs are evenly
            # distributed.
            ramp_frac = 1.0 - 0.5 * (ngroups - self.mingroups) / (ngroups - 1.0)

        # the effective ramp exposure/measurement time, t_eff, is t_tot * (1 - ramp_frac * pix_cr_rate * t_ramp). We use the saturation
        # time as an approximation of the time during which a ramp can get damaged by a CR. This was previously more complex to
        # handle the case where a ramp is both saturated and hit by a CR, but that required a recalculation of the exposure time for
        # the unsaturated groups. We now simplify this (the difference is minimal and would add some complex logic now that the time
        # formulae are detector type dependent). This is simple.
        cr_ngroups = (1.0 - ramp_frac * pix_cr_rate * self.exposure_spec.saturation_time) * ngroups

        return cr_ngroups

    def get_slope_variance(self, rate, unsat):
        # We allow alternate ways to derive a signal from a detector slope,
        # but multiaccum is the default if no mode is provided. Slopemode
        # must be matched with a corresponding slope_variance function.
        if self.slopemode == "target_acq":
            slope_var, slope_rn_var = self.slope_variance_ta(rate, unsat)
        elif self.slopemode == "multiaccum":
            slope_var, slope_rn_var = self.slope_variance(rate, unsat)
        else:
            msg = "{} is not a valid slope mode.".format(self.slopemode)
            raise DataConfigurationError(value=msg)
            
        return slope_var, slope_rn_var

    def _get_groups_before_sat(self, time_to_saturation):
        return ((time_to_saturation / self.exposure_spec.tframe) - self.exposure_spec.nframe - self.exposure_spec.ndrop1) / \
                (self.exposure_spec.nframe + self.exposure_spec.ndrop2) + 1 

class H2RG(Detector_MultiAccum):

    def _get_sample(self):
        """
        H2RG and SiAs detectors use different parameters for slope variance
        (self.slope_variance, self.slope_variance_ta) and readnoise 
        (self.rn_slope_variance) (Issue #2996). Previously, the code tested to 
        see if special sample parameters existed, but now they always do.

        This is the H2RG version.
        """
        m = self.exposure_spec.nframe # This does not include skipped frames!
        tframe = self.exposure_spec.tframe
        return m, tframe

class SiAs(Detector_MultiAccum):

    def _get_sample(self):
        """
        H2RG and SiAs detectors use different parameters for slope variance
        (self.slope_variance, self.slope_variance_ta) and readnoise 
        (self.rn_slope_variance) (Issue #2996). Previously, the code tested to 
        see if special sample parameters existed, but now they always do.

        This is the SiAs version.
        """
        m = self.exposure_spec.nsample_total
        tframe = self.exposure_spec.tsample
        return m, tframe

class H4RG(H2RG):
    """
    Detector properties for H4RG detectors (Roman WFI).
    """
    def set_excess(self, flag):
        # this function turns excess noise on and off. H4RG detectors do not use excess noise parameters. Warn
        # if a user tries to turn it on.
        if flag:
            warnings.warn(f"{type(self).__name__} detectors do not have excess noise parameters, they cannot be turned on.")

#------------ Single accum ------------------

class Detector_SingleAccum(Detector):

    def set_excess(self, flag):
        # this function turns excess noise on and off. SingleAccum detectors do not use H2RG-style excess
        # noise parameters. Warn if a user tries to turn them on.
        if flag:
            warnings.warn(f"{type(self).__name__} detectors do not have excess noise parameters, they cannot be turned on.")

    def postflash_as_rate(self):
        """
        Postflash will not always be defined (and is only valid for some CCDs)

        Postflash must be a positive integer number of electrons.

        It's not really a rate, but to add it to the target flux we must treat its contribution as one, which
        means dividing it by the single exposure time - the flash happens before every independent exposure.
        """
        if (("postflash" in self.input_detector) and not self.__dict__.get("has_postflash", False)):
            raise EngineInputError(f"Detector does not support postflash.")
        postflash = self.input_detector.get("postflash", 0)
        if (not isinstance(postflash, int)) or (postflash < 0):
            raise EngineInputError(f"Postflash value {postflash} invalid. Must be positive integer.")

        # now convert to rate.
        postflash_rate = postflash / self.exposure_spec.exposure_time
        return postflash_rate

    def get_before_sat(self, rate):
        """
        This method returns a map of the maximum time each pixel can be exposed before saturation.

        This is considered the equivalent of the multiaccum groups-before-saturation, and serves
        a similar purpose: to tell the user the limits of what they can configure the detector to 
        do before saturation.

        Parameters
        ----------
        slope: ndarray
            The measured slope of the ramp (ie, the rate) per pixel
        """
        saturation_fraction = self.get_saturation_fraction(rate)

        time_before_sat = self.exposure_spec.saturation_time/saturation_fraction

        return time_before_sat

    def get_saturation_fraction(self, slope):
        """
        This method returns a map of the fraction of saturation that each pixel reaches, given the current
        exposure setup.

        Parameters
        ----------
        slope: ndarray
            The measured slope of the ramp (ie, the rate) per pixel

        """
        if isinstance(slope, dict):
            f = self.exposure_spec.saturation_time / self.fullwell
            return f * np.ones(shape=slope['fp_pix'].shape)
        else:
            # postflash is a single additive term taken at the end of the exposure,
            # not a rate. Thus, we need to determine the time for the signal, background,
            # and dark current to fill the REST of the well.
            if hasattr(self, "has_postflash") and "postflash" in self.input_detector:
                slope = slope - self.postflash_as_rate()
                fullwell = self.fullwell-self.input_detector["postflash"]
            else:
                fullwell = self.fullwell
            slope_array = slope.clip(MIN_CLIP, np.max([MIN_CLIP, np.max(slope)]))
            return self.exposure_spec.saturation_time / (fullwell / slope_array)

    def _get_saturation_mask(self, rate):
        """
        Compute a numpy array indicating pixels with (full) saturation (2), and no saturation (0).

        Parameters
        ----------
        rate: None or 2D np.ndarray
            Detector plane rate image used to build saturation map from

        Returns
        -------
        mask: 2D np.ndarray
            Saturation mask image
        """
        saturation_mask = np.zeros(rate.shape)

        saturation_fraction = self.get_saturation_fraction(rate)

        saturation_mask[(saturation_fraction > 1)] = 2

        return saturation_mask

    def get_unsaturated(self, rate, full_saturation=None):

        saturation_fraction = self.get_saturation_fraction(rate)

        unsat = ma.masked_greater(saturation_fraction, 1)

        return unsat

    def slope_variance(self, rate, unsat_ngroups):
        """
        Calculate the variance of a specified SingleAccum slope.
        Inputs
        ------
        rate: ndarray
            The measured slope of the ramp (ie, the rate) per pixel
            This can be a one-dimensional wavelength spectrum (for a 1D ETC)
            or a three-dimensional cube ([wave,x,y] for a 3D ETC).
        Returns
        -------
        slope_var: ndarray
            Variance associated with the input slope.
        slope_rn_var: ndarray
            The associated variance of the readnoise only
        """

        # The noise calculation depends on the pixel rate BEFORE IPC convolution. fp_pix_variance takes that
        # pre-IPC rate and scales it by the quantum yield and Fano factor to get the per-pixel variance in the
        # electron rate.
        
        # TODO: Determine if further changes to this equation need to be made. (JETC-2121) The reverse
        # calculation equation, derived from this noise computation, does not include Fano noise, nor does the
        # PyETC version. If we use a signal that includes fano noise, the reverse calculation does not reverse
        # the SNR calculation by a factor of more than 20%. Without it, the calculation is discrepant by ~1%.
        # More to the point, the Fano noise trends negative at longer wavelengths, which is almost certainly
        # unphysical. PyETC does not include Fano noise even though it's supposed to apply to detector
        # effects. It may yet be necessary to alter the Fano factor (set to 0) and quantum yield calculations
        # so that we CAN use an fp_pix_variance property that is never run through IPC convolution.

        variance_per_pix = rate['fp_pix'] / self.exposure_spec.measurement_time

        # SingleAccum detectors are read once, but signal variance and dark current are time-dependent.
        slope_rn_var = self.rn_variance() * np.ones_like(variance_per_pix) / self.exposure_spec.measurement_time**2
        slope_var = variance_per_pix + slope_rn_var

        # Noise is only valid when the pixel isn't saturated - on MultiAccum, that means operating on just the
        # unsaturated groups. In SingleAccum it means masking out the saturated pixels - they have no noise.
        # The default fill value for masked arrays is a finite number, so convert to ndarrays, and fill with
        # NaNs to make sure missing values are interpreted as truly undefined downstream.

        mask = self._get_saturation_mask(rate['fp_pix'])

        # Commenting this out so that 1D and scalar values can still be calculated for
        # saturated HST observations. This will not have an effect on the saturation maps,
        # the display of the 2D saturation, or the saturation warnings. It will, however,
        # prevent 1D plots from showing NaN in saturated areas. Because this change is
        # only to SingleAccum noise calculations, it will not affect JWST or Roman.
        # slope_var[mask==2] = np.nan 
        # slope_rn_var[mask==2] = np.nan

        return np.asarray(slope_var, dtype=np.float32), np.asarray(slope_rn_var, dtype=np.float32)

    def rn_variance(self):
        """
        Returns the variance due to read noise only.

        Returns
        -------
        rn: float
           Variance associated with the read noise only.

        """
        return self.rn**2 * self.exposure_spec.nsplit

    def calc_cr_loss(self, unsat_map, pix_cr_rate):
        """
        Calculate the effective loss of exposure time due to cosmic rays. This uses the cosmic ray (CR) event rate
        that is contained within the telescope configuration to calculate the odds of a cosmic rays
        hitting a pixel. Pixels that are hit by cosmic rays are assumed to have ramps that are valid before the
        CR, but not after. The exposure specification is used to adjust the expectation value of the exposure
        time (i.e. ngroups) for a CR-truncated ramp.

        See discussion and links in:

        - https://confluence.stsci.edu/pages/viewpage.action?spaceKey=JWST&title=2014-11-10+Effect+of+CRs+on+SNR
        - Robberto (2010) Technical Report JWST-STScI-001928

        for more information on implementation and the numbers used.

        There are singleaccum versions of the function im the above technical report, but fundamentally cosmic ray
        loss in SingleAccum detectors cannot be treated as statistical loss of exposure time: you either lose the
        entire pixel or you don't. Correspondingly, either the user can interpolate over (or coadd, otherwise 
        remove) the cosmic rays, in which case the full exposure time applies; or they can't, in which case the 
        observation is useless to them. 

        Parameters
        ----------
        unsat_map: int
            Map of unsaturated pixels

        pix_cr_rate: float
            Number of cosmic ray events per pixel area per second

        Returns
        -------
        cr_unsat: float
            This is the map of unsaturated pixels, scaled by the mean loss of time due to cosmic ray events.
        """

        # # if for some reason (e.g. using single read noise model) ngroup is less than the minimum needed
        # # to get a good ramp fit, then the whole ramp is lost upon a CR event
        # ramp_frac = 1.0

        # # the effective ramp exposure/measurement time, t_eff, is t_tot * (1 - ramp_frac * pix_cr_rate * t_ramp). We use the saturation
        # # time as an approximation of the time during which a ramp can get damaged by a CR. This was previously more complex to
        # # handle the case where a ramp is both saturated and hit by a CR, but that required a recalculation of the exposure time for
        # # the unsaturated groups. We now simplify this (the difference is minimal and would add some complex logic now that the time
        # # formulae are detector type dependent). This is simple.
        # cr_unsat = (1.0 - ramp_frac * pix_cr_rate * self.exposure_spec.saturation_time) * unsat_map

        return unsat_map

    def compute_scattering_factor(self, wave):
        """
        Compute the echelle scattered light parameter.

        Parameters
        ----------
        wave: float, numpy.ndarray
            Wavelength value(s) for which the scattering is desired.

        Returns
        -------
        scatter_factor

        """
        wave_scatter, mult_scatter = self.echelle_scattering
        scattering_val = np.interp(wave, wave_scatter, mult_scatter)
        scatter_factor = (1. / scattering_val - 1.)

        return scatter_factor

    def _get_variance_fudge(self, wave):
        """
        Some instruments (e.g. MIRI) have measured noise properties that are not well explained
        or modeled by Pandeia's current noise model. This method returns the multiplicative fudge factor
        that is to be applied to the pixel rate variance to match IDT expectations. The fudge factor can be
        chromatic so needs to accept an array of wavelengths to interpolate over. By default this returns
        a 1's array the same length as wave scaled by the 'var_fudge' scalar value.

        For single accum detectors, we keep this as identically 1.0 for now.

        Parameters
        ----------
        wave: 1D numpy.ndarray
            Wavelengths to interpolate chromatic variance over
        """
        return np.ones(len(wave))

    def set_time(self, new_time):
        self.exposure_spec.measurement_time = new_time
        self.exposure_spec.exposure_time = new_time / self.exposure_spec.nsplit
        self.exposure_spec.saturation_time = new_time / self.exposure_spec.nsplit
        self.exposure_spec.total_exposure_time = new_time

    def scale_exposure_time(self, extracted, calc_input):
        """
        Scale the exposure time to match the SNR we want.

        To do this requires a FULL understanding of the SingleAccum noise calculation done in the 
        Strategy and Report classes, because we need to implement its reverse here.

        This is only valid for SingleAccum calculations, where the function is:
        SNR = SR / sqrt(SR/T + BR/T + DR/T + ((Rn**2 + P) * nreads * npix)/T**2)
        and
        SR = Source Rate (electrons/s)
        BR = Background Rate (electrons/s)
        DR = Dark Current Rate (electrons/s)
        Rn = Readnoise (electrons)
        P = Postflash (electrons)
        nreads = Number of independent reads
        npix = Pixels over which Readnoise and Postflash are given (as they are per-pixel 
         values still)
        T = Time (s)
        (note the differences between this and the pyetc time formula - pyetc deals in counts; 
        Pandeia deals with rates)

        Per JETC-2065, solving for T (with an expected input SNR) yields this equation:
        0 = (Rn**2 + P)*npix*nreads + (SR + BR + DR) * T - (SR/SNR)**2 * T**2
        Which is a quadratic equation, solveable in the usual way.

        The complications to this (JETC-2121) are as follows:
        1. The Strategy outputs the extracted flux rate (SR) composed of just the source rate 
           itself, and a flux_plus_bg rate composed of the source, background, dark current, WFC3 
           thermal background, and postflash (SBDPR) - everything but readnoise, which only ever
           appears as a noise term. 
           - Postflash is included as a rate in SBDPR so that it appears in the 2D signal 
             products, but it is fundamentally not a rate and must be removed from SBDPR 
             (making SBDR) so it can be re-added correctly.
           - The other components (all of SBDR) are genuine rates.
           - The noise may include an ADDITIONAL term of STIS Echelle scattering added AFTER 
             Strategy.extract(), which will need to be accounted for. It is also a rate.
        2. The correlated noise extraction in the Strategy extracts the signal and noise through 
           (effectively) different apertures. If M is the mask array, the signal extraction is 
           M * SR and and the noise extraction is M * NR * M^T (see Strategy._error_sum), which 
           is equivalent to extracting the noise through M**2, which is smaller than M.
           Example:
           If this is the mask: 0    0.5  0
                                0.5  1    0.5
                                0    0.5  0       It covers 3 pixels (0.5 + 0.5 + 1 + 0.5 + 0.5)
           Then the noise aperture is the above matrix squared, or this:
                                0    0.25 0
                                0.25  1    0.25
                                0    0.25 0      It covers 2 pixels (0.25 + 0.25 + 1 + 0.25 + 0.25)
           This code computes the required noise aperture size (npix_noise) and mask correction 
           factors (mask_correction), which would be 0.66667 in my simple example here.
           - Signal is NOT spatially uniform (we cannot guess what the source flux present in those
             edge pixels is), so we really do need to use an actual extraction through the noise 
             aperture (extracted['extracted_source_noise']). We also have to remove the extracted 
             signal from the extracted flux_plus_bg_rate. (SBDR-SR = BDR)
           - Background, dark current, and thermal are all spatially uniform, and thus can be 
             handled by multiplying BDR by the mask_correction factor.
           - Postflash and Readnoise are also spatially uniform and given per pixel, and need to 
             be multiplied by the number of pixels in the noise aperture.
           - We do, however, still need the real extracted signal rate for the SR/SNR term.

        Parameters
        ----------
        extracted: dict
            Dictionary of output products produced by Strategy.extract
        calc_input: dict
            Dictionary of user inputs

        Returns
        -------
        time: float
            The exposure time to scale to.

        """
        wave_index = 0
        wave_pix = extracted['wavelength'] # at worst, this will be backwards from the actual wavelength grid
                                           # but we explicitly use min and max, so this should be fine.
        if len(wave_pix) > 1:
            wref = (wave_pix.max()+wave_pix.min())/2.
            user_wave = calc_input['strategy']['reference_wavelength']
            if user_wave is not None:
                if user_wave >= wave_pix.min() or user_wave <= wave_pix.max():
                    wref = user_wave
                # the Report version of this conditional produces a warning here (bad_waveref), but we won't 
                # bother now and let the Report produce it later.

            sbdpr = self.to_resels_scalar(wave_pix, extracted['extracted_flux_plus_bg_all'], wref) # (source + background + dark + postflash) rate
            sr = self.to_resels_scalar(wave_pix, extracted['extracted_flux'], wref) # source rate
            ns = self.to_resels_scalar(wave_pix, extracted['extracted_source_noise'], wref)
            if wref >= wave_pix[-2]:
                mask = extracted['masks_products'][-1]
            elif wref <= wave_pix[1]:
                mask = extracted['masks_products'][0]
            else:
                idx = bisect.bisect(wave_pix, wref)
                mask_0 = extracted['masks_products'][idx]
                mask_1 = extracted['masks_products'][idx+1]
                mask = mask_0 + (mask_1 - mask_0) * (wref - wave_pix[idx-1]) / (wave_pix[idx] - wave_pix[idx-1])
                
            # Echelle scattering, where it exists, must be accounted for in our noise formula
            if hasattr(self, "echelle_scattering"):
                scatter = self.compute_scattering_factor(wref)
            else:
                scatter = 0
        else:
            sbdpr = extracted['extracted_flux_plus_bg_all'][0] # (source + background + dark + postflash) rate
            sr = extracted['extracted_flux'][0] # source rate
            ns = extracted['extracted_source_noise'][0] # source rate through noise aperture
            mask = extracted['masks_products'][0]
            scatter = 0
            
        # The SNR we're shooting for, NOT the one we achieved
        snr = calc_input['configuration']['detector']['snr']

        # JETC-3020: Correction for HST spectroscopic calculations. The mask size we use
        # in the reverse calculation equation needs to be (pixels-per-resel) wide, because
        # the values it's using in the computation are in resels.
        pixels_per_resel = self.pixels_per_resel
        # this check excludes imaging modes (JETC-3020)
        if len(wave_pix) == 1:
            pixels_per_resel = 1

        # When background_subtraction is True, there will be negative pixels
        npix = np.sum(np.abs(mask)) * pixels_per_resel
        # the noise is computed over effectively fewer pixels, due to the fact that the
        # noise is squared and the edges of the mask are only partially transparent. But
        # that only applies in the cross-dispersion direction, so multiplying that value
        # by the width in unit pixels is appropriate.
        npix_noise = np.sum(mask**2) * pixels_per_resel

        mask_correction = npix_noise/npix
        # postflash is in the plus_background rate array but it is fundamentally not a
        # time-dependent term.
        postflash = self.input_detector.get("postflash", 0)
        sbdr = sbdpr - self.postflash_as_rate()*npix
        # we also need to remove the signal rate, because it is NOT a uniform term.
        bdr = (sbdr - sr) * mask_correction

        delta = (ns + bdr + (ns * scatter))**2 + 4 * (sr / snr)**2 * (self.rn**2 + postflash) * npix_noise * self.exposure_spec.nsplit

        time = ((ns + bdr + (ns * scatter)) + np.sqrt(delta)) / (2 * (sr / snr)**2) / pixels_per_resel

        return time
class CCD(Detector_SingleAccum):
    pass

class H1R(CCD):
    pass

class MAMA(Detector_SingleAccum):

    def __init__(self, config, webapp=False):
        """
        Sets the read noise correlation flag to False, instead of resorting to a
        definition in a config file (config files for MAMA detectors should not
        refer to a non-existent concept of read noise)
        """
        Detector_SingleAccum.__init__(self, config, webapp=False)

        self.rn_correlation = False

    def set_saturation(self, toggle):
        # MAMA detector has no concept of saturation, so this is irrelevant.
        pass

    def set_readnoise(self, flag):
        # this function turns readnoise on and off, but MAMAs do not have readnoise to turn on. Warn if a user
        # tries to turn it on.
        if flag:
            warnings.warn(f"{type(self).__name__} detectors do not have readnoise, it cannot be turned on.")

    def get_saturation_fraction(self, rate):
        """
        This method returns a map of the fraction of pixels that exceed
        the bright pixel rate threshold, given the current exposure setup.

        (we are reusing the 'saturation' word in the method's name. Maybe this
        should be revisited further down the road? For now, 'saturation' means,
        in the MAMA context, 'exceeds bright pixel threshold').

        Parameters
        ----------
        rate: ndarray
            The measured rate per pixel

        """
        arate = rate
        if isinstance(rate, dict):
            arate = rate['fp_pix']

        return np.zeros_like(arate)

    def rn_variance(self):
        """
        Returns the variance due to read noise only, which is identically zero
        for photon-counting sensors.

        Returns
        -------
        0.0: float
           The variance associated with the read noise on a MAMA detector is zero.

        """
        if self.rn != 0:
            warnings.warn(f"{type(self).__name__} detectors do not have readnoise, but it was set to {self.rn} e-/s. Setting to zero...")
            self.rn = 0
        return 0.0

class XDL(MAMA):
    pass
