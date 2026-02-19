# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .telescope import Telescope
from .instrument import Instrument
from .custom_exceptions import DataError
from .constants import DEFAULT_DISPERSION, PANDEIA_WAVEUNITS, PANDEIA_FLUXUNITS
import astropy.units as u
from astropy.table import Table
import synphot as syn
import stsynphot as st
import numpy as np
from numpy.polynomial import polynomial


class HST(Telescope):

    def get_ote_eff(self, wave):
        """
        Temporary function; allow stsynphot to handle all of the graph table functionality in one go.
        """
        return np.ones_like(wave)



class HSTInstrument(Instrument):

    """
    Generic HST Instrument class
    """
    def __init__(self, mode=None, config={}, **kwargs):
        telescope = HST()
        self.instrument_pars = {}
        # TODO HST has in general single accum detectors for which ngroup and nint do not make sense.
        self.instrument_pars['detector'] = ["nsplit"]
        self.instrument_pars['instrument'] = ["aperture", "disperser", "detector", "filter", "instrument", "mode"]
        self.api_parameters = list(self.instrument_pars.keys())

        # these are required for calculation, but ok to live with config file defaults
        self.api_ignore = ['dynamic_scene', 'max_scene_size', 'scene_size']
        Instrument.__init__(self, telescope=telescope, mode=mode, config=config, **kwargs)
        # avoid doing the complex obsmode calculations multiple times
        self.obsmode = self.get_obsmode()
        self.rpower, self.dlds, self.dispwave = self.get_dispersion_products()

    def get_obsmode(self):
        if "imagingacq" in self.instrument['mode']:
            obsmode = self._obsmode_ta()
        elif "imaging" in self.instrument['mode']:
            obsmode = self._obsmode_imaging()
        elif "specacq" in self.instrument['mode']:
            obsmode = self._obsmode_specacq()
        elif "spec" in self.instrument['mode']:
            obsmode = self._obsmode_spec()
        elif "echelle" in self.instrument['mode']:
            obsmode = self._obsmode_spec()
            
        if "mjd" in self.instrument:
            mjd = self.instrument['mjd']
        else:
            mjd = self.telescope.mjd
        obsmode += ",mjd#{}".format(mjd)

        return obsmode

    def _obsmode_imaging(self):
        """
        Temporary function for basic imaging obsmodes, to be replaced as we work modes.
        These are likely not correct and not complete, but they should be functional.
        """
        return "{},{},{}".format(self.instrument['instrument'], self.instrument['detector'], self.instrument['filter'])

    def _obsmode_spec(self):
        """
        Temporary function for basic spectroscopic obsmodes, to be replaced as we work modes.
        These are likely not correct and not complete, but they should be functional.
        """
        if "cenwave" in self.instrument:
            return "{},{},{},{}".format(self.instrument['instrument'], self.instrument['detector'], self.instrument['disperser'], self.instrument['cenwave'])
        else:
            return "{},{},{}".format(self.instrument['instrument'], self.instrument['detector'], self.instrument['disperser'])

    def _obsmode_specacq(self):
        """
        Base class defaults to spec mode
        """
        return self._obsmode_spec()

    def _obsmode_ta(self):
        """
        Base class defaults to imaging mode
        """
        return self._obsmode_imaging()

    def get_dispersion_products(self):
        """
        For HST we don't have separate files, we have the TRDS graph table
        and the default wavelength range being output
        """

        band = st.band(self.obsmode)
        dispwave = band.binset.to_value(u.micron)

        # code from pandeia_data/devtools/create_MIRI_LRS_dispersion.py
        dlds_int = np.gradient(dispwave)
        dlds_coeffs = polynomial.polyfit(dispwave,dlds_int,7)
        dlds = polynomial.polyval(dispwave, dlds_coeffs)

        r = dispwave/dlds
        return r, dlds, dispwave

    def get_dispersion(self, wave):
        """
        Return dispersion

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate/trim dispersion data onto

        Returns
        -------
        dispersion: numpy.ndarray or float
            Dispersion as a function of wave
        """
        disperser = self._get_disperser()
        if disperser is not None:
            dlds = self.dlds
            dlds = self._interp_refdata(wave, dlds, "dlds", default=DEFAULT_DISPERSION)
        else:
            dlds = 0.0
        return dlds

    def get_resolving_power(self, wave):
        """
        Return the resolving power

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate resolving power onto

        Returns
        -------
        rpower: numpy.ndarray or None
            Resolving power as a function of wave
        """
        disperser = self._get_disperser()
        if disperser is not None:
            rpower = self.rpower
            rpower = self._interp_refdata(wave, rpower, "R", default=wave/(DEFAULT_DISPERSION*2))
        else:
            rpower = None
        return rpower

    def get_wave_pix(self):
        """
        Get the wavelength vector to convert pixel position to wavelength
        For HST, stsynphot and the graph table have already done this for us.

        Returns
        -------
        wavepix: numpy.ndarray
            Wavelength vector mapping pixels to wavelengths
        """
        return self.dispwave

    def get_detector_pixels(self, wave):
        """
        Read in detector pixel positions for each wavelength in wave_pix

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate/trim pixel position data onto

        Returns
        -------
        pixels: numpy.ndarray or None
            Detector pixel positions corresponding to each element of wave
        """
        disperser = self._get_disperser()
        if disperser is not None:
            pixels = np.arange(len(wave))
        else:
            pixels = None
        return pixels

    def get_wave_blaze(self):
        """
        Return the wavelength vector used in the grating efficiency (blaze) file

        Returns
        -------
        wave_blaze: numpy.ndarray
            Wavelength vector from the grating efficiency file
        """
        return self.dispwave


    def _interp_refdata(self, wave, data, name, default=None):
        """
        Read in requested reference file and return requested data interpolated onto wave

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength array onto which the reference data is to be interpolated
        data: numpy.ndarray
            Key to get filename from self.paths
        name: string
            Name of the reference data; only used for the error message
        default: None or float
            Default value to return in case of missing file or column

        Returns
        -------
        ref: numpy.ndarray or float
            If file exists, return reference_data(wave). Else return default.
        """
        ref = None
        if data is not None:
            ref = np.interp(wave, self.dispwave, data)
        if ref is None and default is None:
            msg = "No reference data found and no default value supplied for %s." % (name)
            raise DataError(value=msg)
        elif ref is None and default is not None:
            ref = np.ones_like(wave)*default
        return ref

    def _get_cdbs(self, wave, key):
        """
        HST data is available in the $PYSYN_CBDS trees, so it should be loaded differently.

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength array onto which the throughput curve is to be interpolated
        key: str
            Key to get filename from the $PYSYN_CDBS tree

        Returns
        -------
        eff: numpy.ndarray or float
            If ref file exists, return efficiency(wave), else return 1.0
        """
        if 'None' not in key:
            bp = st.spectrum.band(key)
            eff = bp(wave*u.micron).value #don't want to pass the Quantity
        else:
            # If it's explicitly None, there is no element; pass through 1.0
            eff = np.ones_like(wave)*1.0
        return eff

    def get_filter_eff(self, wave):
        """
        Stub replacement to allow stsynphot to handle all of the graph table functionality in one go.
        """
        return np.ones_like(wave)

    def get_disperser_eff(self, wave):
        """
        Stub replacement to allow stsynphot to handle all of the graph table functionality in one go.
        """
        return np.ones_like(wave)

    def get_detector_qe(self, wave):
        """
        Stub replacement to allow stsynphot to handle all of the graph table functionality in one go.
        """
        return np.ones_like(wave)

    def get_internal_eff(self, wave, full_throughput=False):
        """
        This is the only temporary function that will actually do anything. It relies on
        stsynphot for graph table functionality. This does not run
        Instrument._get_throughput(), which would limit arrays to 32 bit. HST's
        high-resolution modes require 64-bit arrays.
        
        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate throughput onto
        full_throughput: bool
            Dummy value that exists so the STIS-specific get_internal_eff can include slit
            losses.

        Returns
        -------
        eff: numpy.ndarray or float
            Disperser efficiency as a function of wave
        """
        bp = st.spectrum.band(self.obsmode)
        eff = bp(wave*u.micron).value #don't want to pass the Quantity

        return eff

    def get_quantum_yield(self,wave):
        """
        Compute detector quantum yield if the critical wavelength is defined. If not, 
        return unity (trivial yield).

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate throughput onto

        Returns
        -------
        quantum_yield: numpy.ndarray or float
            Quantum efficiency as a function of wave
        """
        if hasattr(self.the_detector, "critical_wavelength"):
            q_yield = self.the_detector.critical_wavelength / wave
        else:
            q_yield = 1
        fano_factor = (3.0 * q_yield - q_yield**2. - 2.) / q_yield

        return q_yield, fano_factor

class COS(HSTInstrument):

    """
    Special methods unique to HST COS
    """
    def __init__(self, mode=None, config={}, webapp=False, **kwargs):
        """
        Like JWST NIRSpec, COS needs to have the chip segments/stripes specially loaded
        """
        HSTInstrument.__init__(self, mode=mode, config=config, webapp=webapp, **kwargs)
        slit = self.instrument.get('slit', None)
        # Even COS imaging modes have slits, so a slit is not indicative of an issue.
        if self.projection_type != "image":
            try:
                slit_config = self.slit_config[slit]
            except KeyError as e:
                msg = "Configuration for slit {} not specified".format(slit)
                raise DataError(value=msg)
            self.segments = self.read_config_param(slit_config, "range")
        self._loadpsfs()

    def get_wave_range(self):
        if self.projection_type != "image":
            # get the wavelength range from the segments configuration
            disperser = self.instrument['disperser']
            cenwave = self.instrument['cenwave']
            segments = self.segments[disperser][cenwave]
            wmin = None
            wmax = None
            for segment in segments:
                if wmin == None or segment["wmin"] < wmin:
                    wmin = segment["wmin"]
                if wmax == None or segment["wmax"] > wmax:
                    wmax = segment["wmax"]
        else:
            aperture = self.instrument['aperture']
            slit = self.instrument['slit']
            wmin = self.range[aperture][slit]["wmin"]
            wmax = self.range[aperture][slit]["wmax"]

        range_dict = {'wmin': wmin, 'wmax': wmax}
        return range_dict

    def _get_psf_key(self):
        """
        The COS PSFs differ greatly by focus position, and we thus need to 
        know lifetime position (currently only LP4), disperser, and cenwave.
        """
        psf_key = self.instrument['aperture']

        if self.instrument['mode'] in ["fuv_spec","fuv_specacq", "nuv_specacq"]:
            psf_key = "{}{}{}".format(self.instrument['aperture'], self.instrument['disperser'], self.instrument['cenwave'])
        elif self.instrument['mode'] in ["nuv_imaging","nuv_imagingacq"]:
            psf_key = self.instrument['aperture']+self.instrument['slit']

        return psf_key

    def _obsmode_imaging(self):
        items = ['cos',self.instrument['aperture'],self.instrument['detector'], self.instrument['slit']]
        obsmode = ",".join(items)
        return obsmode

    def _obsmode_spec(self):
        items = ['cos',self.instrument['detector'],self.instrument['disperser'],str(self.instrument['cenwave']),self.instrument['slit']]
        obsmode = ",".join(items)
        return obsmode

    # specacq mode returns the same obsmode as spec - rely on base class directly
    def get_segment_indicies(self, wave):
        """
        Generic code to get the boundaries of a chip/stripe/segment in pixel coordinates, from our segment
        definitions.

        Parameters
        ----------
        wave: np.ndarray
            The wavelength array of the projected grid

        Returns
        -------
        segment_indices: list
            A list of dictionaries defining the segments in pixel coordinates.
        """
        disperser = self.instrument['disperser']
        cenwave = self.instrument['cenwave']

        segments = self.segments[disperser][cenwave]
        segment_indicies = []

        # loop over each segment
        for i in range(len(segments)):

            segname = segments[i]['name']

            # compute segment limits in X direction
            wmin = segments[i]['wmin']
            wmax = segments[i]['wmax']

            wmin_ind = np.searchsorted(wave,wmin, side="right")
            wmax_ind = np.searchsorted(wave,wmax, side="left")

            segment_dict = {"segname": segname, "wmin_ind": wmin_ind, "wmax_ind": wmax_ind}
            segment_indicies.append(segment_dict)

        return segment_indicies

    def get_chip_dimensions(self, wave):
        """
        COS FUV detector is comprised of two chips that have their own dark current, and require their own
        health and safety outputs; the spectrum is produced as a single array with a gap.

        Thus, we have both the complete dimensions of the chip (in pixel space) and the portion of the
        computed spectrum on that chip to consider.

        The COS NUV detector, on the other hand, is actually a single chip, but the optics split the spectrum
        into three stripes that are imaged separately over three different locations on the chip. We use this
        method then to create 3 "virtual segments" that help code downstream in its task of spliting up the
        spectrum into its three separate stripes. This has to be done so extractions and reporting can be
        performed on a per-stripe basis.

        In all of these, 
        
        - "name" is the detector name, 
        - "size" is the size of the section that collects dark current 
        - "bounds" is a dictionary containing the slices that will split up the flux (JUST the signal)
        into the appropriate pieces.

        background treatment is complex - if there is no slit, background will be assumed to cover the entire
        chip ("size"). If there IS a slit, if the slit is taller than the 2D detector image the background
        will be scaled to match. If it's not taller than the 2D image, we just assume the background array
        contains the entirety of the background flux.


        """
        nx = self.the_detector.pixels_x
        ny = self.the_detector.pixels_y

        if self.projection_type != "image":
            segment_indicies = self.get_segment_indicies(wave)
            dims = []
            if hasattr(self.the_detector, 'segnames'):
                for segment in segment_indicies:
                    dims.append({"name": segment["segname"], "size": {"x": nx, "y": ny},
                        "bounds": {"x": slice(segment["wmin_ind"],segment["wmax_ind"]), "y": slice(-1)}})

            elif hasattr(self.the_detector, 'stripenames'):
                stripe_ysize = int(ny / 3)
                for segment in segment_indicies:
                    dims.append({"name": segment["segname"], "size": {"x": nx, "y": stripe_ysize},
                        "bounds": {"x": slice(segment["wmin_ind"],segment["wmax_ind"]), "y": slice(-1)}})

            return dims

        return super().get_chip_dimensions(wave)

    def create_gap_mask(self, wave):
        """
        Use the slit segment configurations and a wavelength vector, wave, to build a masked array that masks
        out the location of the gaps.  Wavelengths between and including both gap endpoints will be masked
        out.

        As a reminder, the slit segments API is, as of 2022.0509:
        slit_config
         slit
          "range" <- self.segments
           subarray
            disperser
             filter or cenwave
              segment or stripe definitions.

        This code will not generate a slit mask unless a.) there is a slit configuration; b.) the actual
        defined slit is not None, and c.) if that slit definition includes a range item.

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to construct mask from

        Returns
        -------
        mask: numpy.ma 1D masked array
            1D array masked at the locations within the configured detector gap and 1.0 elsewhere
        """
        mask = 1.0
        if hasattr(self, "segments"):
            segment_indicies = self.get_segment_indicies(wave)
            
            mask = np.zeros_like(wave)
            # we actually UNmask the non-gap area
            for segment in segment_indicies:
                mask[segment["wmin_ind"]:segment["wmax_ind"]] = 1.0

        return mask


class WFC3(HSTInstrument):

    """
    Special methods unique to HST WFC3
    """
    def __init__(self, mode=None, config={}, webapp=False, **kwargs):

        HSTInstrument.__init__(self, mode=mode, config=config, webapp=webapp, **kwargs)
        self._loadpsfs()

    def _obsmode_imaging(self):
        obsmode = ",".join(["wfc3",self.instrument['detector'],self.instrument['filter']])
        return obsmode

    def _obsmode_spec(self):
        obsmode = ",".join(["wfc3",self.instrument['detector'],self.instrument['disperser']])
        return obsmode

    def add_thermal(self):
        """
        This WFC3-specific functionality adds thermal background if the detector is the WFC3 IR
        detector.

        Per PyETC (engine/instrument.py line 272) thermback produces the integrated thermal 
        background per pixel, and it is to be treated as an additional source of flux akin
        to dark current.

        Returns
        -------
        thermal: float
            Scalar thermal background contribution
        """
        # Thermal is only a relevant parameter for WFC3's IR detector
        if self.instrument['detector'] == "ir":
            bp = st.spectrum.band(self.obsmode)
            # this is built out of integrating a spectrum in PHOTLAM/(arcsec**2) and
            # multiplying by area and (arcsec/pixel)**2, which should yield units of
            # PHOTLAM/(pixel**2) * cm^2.
            # where PHOTLAM is photons/sec/cm^2/Angstrom, and integrating should have 
            # removed the angstrom dependency. The final units should be photons/(sec 
            # pixel^2), which is being called counts/(s pix). That implies counts
            # is a unit of photons/pixel. 
            # we have this conversion: f_lambda = 1.5091905 * (flux / self.wave)
            # where flux is in mJy/pix and f_lambda is in photons/cm^2/sec/micron
            # We convert that value to electrons/s/pix/micron with the 
            # collecting area * ote_eff. And we don't have direct access to the HST 
            # OTA. 
            thermal = bp.thermback().value # counts/(pix sec)
            # We need electrons/(pix sec); which means converting counts to electrons.
            # Current WFC3IR gain is 2.5 e-/ADU, per 
            # https://hst-docs.stsci.edu/wfc3dhb/chapter-7-wfc3-ir-sources-of-error/7-2-gain 
            # (Accessed 2022.0209)
            # But, PyETC doesn't apply any gain, and all my calculations come out nearly 
            # 2.5x too high if I do.
            #thermal = syn.units.convert_flux(effwave*u.micron, bp.thermback(), u.electron/(u.pixel * u.second), area=self.telescope.coll_area)
        else:
            thermal = 0

        return thermal

class ACS(HSTInstrument):

    """
    Special methods unique to HST ACS
    """
    def __init__(self, mode=None, config={}, webapp=False, **kwargs):

        HSTInstrument.__init__(self, mode=mode, config=config, webapp=webapp, **kwargs)
        self._loadpsfs()

    def _obsmode_imaging(self):
        """
        Polarimeter options are already comma-separated and within the filter keyword.
        """
        obsmode = "{},{}".format(self.instrument['instrument'], self.instrument['detector'])
        if self.instrument['filter'] != "clear":
            obsmode += ",{}".format(self.instrument['filter'])
        return obsmode

    def _obsmode_spec(self):
        """
        Polarimeter options are already comma-separated and within the disperser keyword.
        """
        obsmode = "{},{},{}".format(self.instrument['instrument'], self.instrument['detector'], self.instrument['disperser'])
        return obsmode


class STIS(HSTInstrument):

    """
    Special methods unique to HST STIS
    """
    def __init__(self, mode=None, config={}, **kwargs):
        """
        STIS has additional configuration parameters, like fuv_glow_region and dark_level.
        Do the standard HSTInstrument setup, then read in the extra parameters.
        """
        super().__init__(mode, config, **kwargs)

        if 'fuvmama' in self.instrument['detector']:
            fuv_glow_key = 'dark_glow_region_{}'.format(config['detector']['fuv_glow_region'])
            fuv_glow_rate = self.detector_config['fuvmama'][fuv_glow_key]
            self.the_detector.dark_current = self.the_detector.dark_current + fuv_glow_rate
        self._loadpsfs()

    @property
    def projection_type(self):
        """
        Determine the appropriate projection type based on the configured instrument mode

        STIS is special, as its spec modes mix slit (spec) and slitless projections in the same
        mode. The selection is controlled by a "slitless_slits" list in the config.json file

        Returns
        -------
        proj_type: str
            The projection type, currently one of 'slitless', 'slitless_scan', 'spec', 'image', or 'image_scan'.
        """

        proj_type = super().projection_type
        if proj_type in ("spec", "slitless"):
            if self.instrument['slit'] in self.slitless_slits:
                proj_type = "slitless"
        return proj_type

    def _obsmode_imaging(self):
        items = ['stis',self.instrument['detector'], 'mirror']
        if self.instrument['filter'] is not None:
            items.append(self.instrument['filter'])
        obsmode = ",".join(items)
        return obsmode

    def _obsmode_spec(self):
        """
        Much of the PyETC version of this code has to deal with the fact that the web interface
        does not contain the prefix character (i or c). In pandeia, the names of the cenwaves include
        the prefix.
        """
        disperser = self.instrument['disperser']
        detector = self.instrument['detector']
        obsmode = ",".join(['stis', detector, disperser])

        cenwave = self.instrument.get('cenwave',None)
        if cenwave != None:
            obsmode += ",{}".format(cenwave)
        else: #use default central wavelength if there is one
            try:
                cenwave = self.config_constraints['disperser'][disperser]['cenwaves'][0]
                obsmode += ",{}".format(cenwave)
            except (KeyError, IndexError):
                pass
                #otherwise leave it off
        
        slit = self.instrument.get('slit', None)
        if slit is not None:
            obsmode += ",{}".format(slit)

        return obsmode

    def get_internal_eff(self, wave, full_throughput=False):
        """
        Special STIS version of the get_internal_eff function. When a slit has both a
        filter AND a narrow slit, we need to remove the effect of the narrow slit so we
        don't double count it.
        
        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate throughput onto
        full_throughput: bool
            Convenience for STIS Throughput plots, which DO want the slit losses.

        Returns
        -------
        eff: numpy.ndarray or float
            Disperser efficiency as a function of wave
        """
        slit = self.obsmode.split(",")[-2]

        if full_throughput:
            # For throughput plots, where we DO want to see the slit losses.
            bp = st.spectrum.band(self.obsmode)
            eff = bp(wave*u.micron).value # don't want to pass the Quantity
        elif slit in self.slitless_slits:
            # if the slit is slitless (only a filter) we don't need to worry about
            # double-counting light losses and can just use it.
            bp = st.spectrum.band(self.obsmode)
            eff = bp(wave*u.micron).value # don't want to pass the Quantity
        elif slit in self.unfiltered_slits:
            # if this is an unfiltered slit, we have already taken care of the slit losses
            # geometrically in ConvolvedSceneCube and should remove the slit parameter to
            # avoid double-counting.
            split_obsmode = self.obsmode.split(",")
            del split_obsmode[-2]
            obsmode = ",".join(split_obsmode)
            bp = st.spectrum.band(obsmode)
            eff = bp(wave*u.micron).value # don't want to pass the Quantity
        elif slit in self.filtered_slits:
            # if this is a filtered slit, we need to divide out the specific slit loss
            # component because we already handle that geometrically. 
            bp = st.spectrum.band(self.obsmode)

            # find the comp table
            comptable = st.getref()["comptable"]
            tab = st.stio.read_comptable(comptable)
            # find the STIS 52x0.05 slit file in the comp table
            slitfile = tab[[x.strip() == "stis_52x005" for x in tab["COMPNAME"]]]["FILENAME"][0]
            # load the slit bandpass
            slit_bp = syn.spectrum.SpectralElement.from_file(st.stio.irafconvert(slitfile))

            eff = bp(wave*u.micron).value
            slit_eff = slit_bp(wave*u.micron).value
            eff /= slit_eff

        return eff

    def apply_scattering(self, extracted_list):
        """
        Compute and apply the echelle scattered light parameter Will not affect any configuration that does
        not have defined echelle scattering 

        Parameters
        ----------
        wave: numpy.ndarray
            Wavelength vector to interpolate throughput onto
        rate: extracted target rate

        """
        if hasattr(self.the_detector, "echelle_scattering"):
            for idx in range(len(extracted_list)):

                scatter_rate = self.the_detector.compute_scattering_factor(extracted_list[idx]["wavelength"]) * extracted_list[idx]["extracted_source_noise"]

                # at this point, the noise is sqrt(variance), but we need to add this as a new poisson noise
                # term.
                extracted_list[idx]["extracted_noise"] = np.sqrt(extracted_list[idx]["extracted_noise"]**2 + scatter_rate / self.the_detector.exposure_spec.measurement_time)

        return extracted_list

    def get_global_scattering(self):
        """
        Return the global scattering rate, scattered portion as defined in the data
        """
        cenwave = self.instrument.get('cenwave', None)

        # No cenwave or cenwave = all will give no detector scattering, but PyETC doesn't make that option
        # possible.
        scatter_rate = 0
        if hasattr(self.the_detector, "echelle_global"):
            if cenwave in self.the_detector.echelle_global:
                scatter_rate = self.the_detector.echelle_global[cenwave] - 1

        return scatter_rate
