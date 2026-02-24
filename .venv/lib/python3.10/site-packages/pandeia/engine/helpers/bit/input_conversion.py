"""

This file contains simple input validation and type conversion for
the instrument forms.

The input validation checks that you aren't trying to sneak something
by to subvert the system.  For example, after validation, an instrument
name can only ever contain letters.  Even if you use the instrument
name in os.system(), you don't have to worry about the user submitting
an instrument named "; rm -rf / ; " and fooling your program into removing
files from your system.

The entry point is instrument_input_conversion().

All request forms share a single name space.  If an instrument has form
field "xyz", it has the same meaning as "xyz" for every other instrument.
The dict named "converter" contains one function for each input field.
That function converts the input value and raises an exception if the
value doesn't look right or can't be converted to the right type.

In principle, the user never sees an exception because javascript in
the input form will prevent a bad form value being submitted.
(This is not fully implemented. 2009-08-19)

There are several functions defined for use in converter[].  You can use
one of them directly, or you can use a lambda function.  If the expression
gets complicated, you would want to pull it out into a separate function
anyway, but we make extensive use of lambda function to verify lists:

converter = {

    'sample1':  alnum,
            # Tests that the field is an alphanumeric string

    'sample2':  float_range,
            # the field is converted to floating point

    'sample3':  lambda v, n : float_range( v, n, min=1.0, max=10.0 ),
            # the field is floating point with min/max values

    'sample4':  lambda v, n : int_range( v, n, min=0, max=10 ),
            # same thing as integer

    'sample5':  lambda v, n : list_int( v, n, [ 1, 2, 4, 8, 16 ] ),
            # field is integer, but only one of the listed values

    'sample6':  lambda v, n : list_str( v, n, [ "a", "b", "hello" ]),
            # field is string, but only one of the listed values

    'sample7':  declination,
    'sample8':  right_ascension,
            # these return strings because that is what other parts of
            # the system expect.  Should they be something else?

There is a thing at the end of this file that checks that all the needed
fields are listed in converter[].  It runs on import if debugging is
turned on.

"""
import os, re, sys

import pysynphot
from pandeia.engine.helpers.bit.pyetc_util import dynamic_import, log
CFG_DEBUG = log.VERBOSE_FLAG # no need for full web config here, just use this


valid_instrument_name = re.compile('^[a-z0-9]+$')
valid_science_mode = re.compile('^[A-Za-z_]+$')

instrument_modules = { }
################################################################################

class InputConversionError(ValueError):
    """An exception for bad inputs which should be displayed as other than server
    errors.
    """
    pass

class InputConversionHackDetect(Exception):
    '''An exception to raise if the input looks like an attack

    In this case, we just let it bubble up to the user as "internal error"
    without giving any clues.  Developers can see it in the log file.
    '''
    pass

def handle_file_uploads(request, request_dir, upload_list):
    """Check uploaded files and copy to request directory.
    """
    # if there are any uploaded files, store them in request_dir.
    #
    # For an upload filed named 'Abc', the request will contain a key
    # 'Abc' that contains the name the user gave the file, and 'AbcLocal'
    # that contains the base name of the actual file on disk.
    #
    # The file name on disk is the form field name, plus any extension
    # that the user provided IF we recognize the extension.
    # For example, if the form contains "fUploadFile", and the user
    # uploads "xyz.fits", request_dir will contain "fUploadFile.fits".
    #

    # upload_list is a list of the uploaded files that the input verifier
    # saw in the form and considered valid upload names.  Therefore, we
    # know that they are all there.  The input verifier did not check
    # the files, because we can't fully accept the upload until after
    # creating the etc_id.  So, we verify it in handle_upload()
    #
    updates = {}
    for x in upload_list :
        updates[x] = request.FILES[x].name
        # xxxLocal is the file name where we stored the upload for field xxx
        # it is relative to request_dir
        # The actual file name may end with ".fits" or ".txt" or "".
        updates[x+'Local'] = handle_upload(request, request_dir, x)
    return updates


##

# see use of upload_helper in handle_upload just below
def upload_helper( request, fieldname, request_dir ) :

    # the name of the form field will be the name of the file on our disk...
    f = request.FILES[fieldname]
    filename = str(fieldname)

    # ...maybe with .fits or .txt added
    for x in [ '.fits', '.txt' ] :
        if f.name.endswith(x) :
            filename = filename + x
            break

    # collect the data into the file.
    # (ok to do now - we would not be here if it is not a field we expect.)
    fullpath = os.path.join(request_dir, filename )
    where = open(fullpath,"wb+")
    for chunk in f.chunks():
        where.write(chunk)
    where.close()

    #
    return filename, fullpath


def handle_upload(request, request_dir, fieldname ) :
    """Stores an uploaded file from the web form into the data directory.

    request - the django request object

    fieldname - the name of the field in the input form.

    request_dir - the name of the data directory to store the file in; the
        file name will be the same as the fieldname, but with recognized
        extentions copied from the uploaded name.

    recgonized extentions are:
        .fits
        .txt

    Security note:  This function guards against attacks where there
    user selects an arbitrary field name to get around our file system
    limitations.  It only creates files for those uploadable fields
    that we recognize.  Uploads of other files are not stored.

    """
    if fieldname in request.FILES :

        if fieldname == 'fUploadFile' :
            filename, fullpath = upload_helper( request, fieldname, request_dir )
            try:   # Load either a fits file or an ASCII file.
                   # In any case,  let pysynphot figure it out.
                   # If the file loads,  it's valid.  Discard result.
                pysynphot.FileSpectrum(fullpath)
            except Exception as e:
                # It was an invalid file, but we leave it there for the help desk to look at
                # os.remove(fullpath)
                msg = "Invalid format for uploaded file " + repr(str(request.FILES[fieldname].name))

                # provide more detail if we have any
                if isinstance(e,pysynphot.exceptions.TableFormatError):
                    msg = msg + '\n' + '\n'.join(e.args)
                    raise InputConversionError(msg)

                # raise something even if we don't recognize the problem
                raise

        ## add more uploadable file types here
        # if fieldname == 'whatever' :
        #   filename, fullpath = upload_helper( request, fieldname, request_dir )
        #   ...validate the file...

        else :
            # did not save the file
            raise InputConversionHackDetect('Do not know how to validate file upload %s'%fieldname)

        #
        return filename

    return None


###############################################################################
def convert_dictionary( indict, flist=None):
    """ Alternate entry point for converting a vanilla dictionary
    rather than an actual request.  Used during testing.

    indict:  dictionary
    flist:   optional list of files to be treated as upload files
    """
    strdict = {}
    for key, value in list(indict.items()):
        strdict[key] = str(value)
    request = FakeRequest(strdict, flist)
    fdict, upload_list = instrument_input_conversion( request,
                                                      indict['instrument'],
                                                      indict['science_mode'])
    return fdict, upload_list

class FakeRequest(object):
    """Helper class: used to lie to instrument_input_conversion during testing

    This object looks enough like a django request that we can process
    it through the input conversion, but we don't have to make a complete
    django request object.
"""

    def __init__(self, indict, flist):
        self.POST = indict
        if flist is None:
            self.FILES =[]
        else:
            self.FILES = flist


################################################################################
def instrument_input_conversion( request, instrument, science_mode) :

    # Input is
    #   request: the django request, which contains the form fields that
    #       were submitted.  These are the fields to be validated.
    #   instrument, science_mode: name of the instrument/science_mode
    #       that this form is for; we use this information to choose
    #       what fields to validate.
    #
    # Return is a tuple of ( converted_dict, upload_info ), where
    #   converted_dict: is a dictionary of validated and type-converted
    #       form values, indexed by form field name.
    #   upload_info: is data to hand to handle_uploads() when it is time
    #       to extract uploaded files from the request
    #
    # If any of the fields fail validation, we raise InputConversionError().
    #

    # This part duplicates tests that are already implicit in the django
    # urls.py file, but it doesn't cost much to do it again here and it
    # gives us some protection against programmer errors in future
    # changes to urls.py

    if not valid_instrument_name.match(instrument) :
        raise InputConversionHackDetect('Invalid instrument name %s'%instrument)
    if not valid_science_mode.match(science_mode) :
        raise InputConversionHackDetect('Invalid science mode %s'%science_mode)

    # The list of which fields to expect from each form is stored in
    # the instruments package for that instrument.  Construct the
    # name of the relevant module and import it.

    # Note that the existence of the package pyetc.instruments.XXX_YYY is
    # sufficient validation that XXX/YYY is a valid instrument and science mode.

    key = '%s_%s'%(instrument, science_mode)

    if not key in instrument_modules :
        try :
            m = dynamic_import("pandeia.engine.helpers.bit.instruments.hst.%s.web.form_fields_%s_%s" % (
                instrument, instrument, science_mode))
        except ImportError :
            raise InputConversionHackDetect('invalid instrument/science mode: %s %s'%(
                instrument, science_mode))
        instrument_modules[key] = m
    else :
        m = instrument_modules[key]

    # Now we have the list of fields that we expect to find in the form.
    field_list = m.field_list

    # This dictionary is a list of all the input fields that we actually
    # received.  As we recognize each field, we remove it.  At the end,
    # this dictionary is empty.
    original_inputs = { }
    original_inputs.update(request.POST)

    # This is the result dictionary that contains the translated
    # form inputs.
    fdict = { }

    # This is be the list of uploaded files.  We only list the uploaded
    # files that we actually expect for this form.  (If somebody uploads
    # something else, we never retrieve it.)
    upload_list = [ ]

    # a place to collect all the exceptions that this input form raises.
    # You could just raise an exception on the first bad field, but
    # that sometimes makes it difficult for the developers.
    value_error_strings = [ ]

    # Copy/translate all the fields.  field_list is a list of the form
    # fields that we expect from this form.
    for field in field_list :
        if field in request.POST :
            try :
                if not field in converter :
                    # This only happens during development.
                    print("WARNING: field ",field," IS PROPERLY IN FORM, BUT NOT IN etc_web/etc/input_conversion.py")

                elif converter[field] == upload :
                    # Django passes uploaded files in a different field.
                    # Uploaded files are "not in request.POST" according
                    # to the django documentation, but if the user does
                    # not provide a file to be uploaded, the field _does_
                    # occur in request.POST with the value ''
                    value = request.POST[field]
                    if value != '' and value != '-None-' and value != None:
                        raise InputConversionHackDetect(
                            'upload field is not an uploaded file: %s value-%s-'
                            % (field,request.POST[field])
                        )

                elif converter[field] == DISCARD :
                    # DISCARD is the special function for input fields we
                    # should throw away.  If the tests pass with the field
                    # converter set to DISCARD, you can take that field out
                    # of the html for that form.
                    pass

                else :
                    try :
                        # we have to handle both fields that have a plain value, and
                        # fields that have the value encapsulated as a 1-element list.
                        try:
                            value = request.POST[field].pop()
                        except AttributeError:
                            value = request.POST[field]
                        fdict[field] = converter[field](value , field)
                    except :
                        if CFG_DEBUG :
                            sys.stderr.write(
                                "input field failed validation: %s = %s\n"
                                %(field,request.POST[field])
                            )
                            sys.stderr.flush()
                        raise

            # any failure to validate the input value comes here
            except InputConversionError as e :
                value_error_strings.append(str(e))
            except ValueError as e :
                value_error_strings.append(str(e))

            # after processing the field, remove it.  ( If there are extra
            # fields, they will still be in original_inputs at the end. )
            del original_inputs[field]

    # Uploaded files come separately from other fields of the form.
    for field in request.FILES :
        if converter[field] == upload :
            sys.stderr.write("uploaded file: %s\n"%field)
            upload_list.append(field)
        else :
            # bug: instead of saying so, should we just throw "internal error"?
            s = "uploaded file NOT EXPECTED: %s\n"%field
            value_error_strings.append(s)

    sys.stderr.flush()

    # If any original input is left, we have a problem.
    if len(original_inputs) > 0 :
        s = ""
        if CFG_DEBUG :
            for x in original_inputs :
                value_error_strings.append(
                    "UNEXPECTED INPUT during input_conversion: "+str(x)+'='+str(original_inputs[x])+"\n")
                    # Developer note!  If you see this error after adding a new
                    # field to the UI, make sure you have added your new field
                    # name to the correct "form_fields_<inst>_<mode>.py" file !

    # If there were any validation errors, we raise an exception
    # containing the whole list.
    if len(value_error_strings) > 0 :
        raise InputConversionError("\n".join(value_error_strings))

    ###
    # fdict is a dict containing converted/validated values
    # upload_list is a list of the field names for uploaded files
    return ( fdict, upload_list )

#
def upload( value, name ) :
    raise InputConversionHackDetect('this upload() function should not actually get called')

def DISCARD( value, name ) :
    raise InputConversionHackDetect('this DISCARD() function should not actually get called')

##########
#
# Here are various field validators
#

#

def lookup_nice_info(name):
    '''
    look up information to return to the user about the variable to help them understand the error
    '''

    keyword_information={
        'ZodiMult':'Zodiacal light multiplication factor',
        'EarthshineMult': 'Earthshine multiplication factor',
        'fL1_center': 'Line center value in first row',
        'fL1_flux': 'Flux for line center in first row',
        'fL1_fwhm': 'FWHM for line center in first row',
        'fL2_center' : 'Line center value in second row',
        'fL2_flux': 'Flux for line center in second row',
        'fL2_fwhm' : 'FWHM for line center in second row',
        'fL3_center' : 'Line center value in third row',
        'fL3_flux': 'Flux for line center in third row',
        'fL3_fwhm': 'FWHM for line center in third row',
        'extractionRegionUserCircle': 'Point source extraction region at user radius',
        'extractionRegionUserCircle_min': 'Point source extraction region minimum radius',
        'extractionRegionUserCircle_max': 'Point source extraction region maximum radius',
        'extractionRegionExtendedUserCircle': 'Extended source extraction region at user radius',
        'extractionRegionExtendedUserCircle_min': 'Extended source extraction region minimum radius',
        'extractionRegionExtendedUserCircle_max': 'Extended source extraction region maximum radius',
        'extractionRegionVariableCircle': 'Point source extraction region of x% of light',
        'fbbtemp': 'Black Body Temperature',
        'febv': 'E(B-V)',
        'crsplit' : 'CR-Split ("Frames" or "# Indep. Exposures" for WFC3) \n',

        }

    try:
       newname=keyword_information[name]
    except KeyError:
        newname=name
    return newname


re_alnum = re.compile('^[A-Za-z0-9_]*$')

def alnum( value, name ) :
    if re_alnum.match(value) :
        # str() to de-unicode it
        return str(value)
    raise InputConversionError('%s not alphanumeric'%name)

re_al_dot_num = re.compile('^[A-Za-z0-9._]*$')

def al_dot_num( value, name ) :
    if re_al_dot_num.match(value) :
        # str() to de-unicode it
        return str(value)
    raise InputConversionError('%s not valid'%name)


#
re_for_check = { }

def re_check(value, name, regex ) :
    if not regex in re_for_check :
        re_for_check[regex] = re.compile(regex)
    if re_for_check[regex].match(value) :
        # str() to de-unicode it
        return str(value)
    if name == 'fOtherFile':
        raise InputConversionError("Wrong file name: \"" + value +
                        "\"  -  Check spelling of file name in \"Other HST Spectrum\" input field.")
    else:
        raise InputConversionError('%s value %s not matching regex %s'%(name,value,regex))

# WARNING:  overriding builtin functions min() and max() as parameters
def float_range( value, name, min=None, max=None ) :

    nice_name=lookup_nice_info(name)
    try :
        v = float(value.replace(' ',''))
    except :
        raise InputConversionError('%s (%s) is not a floating point (%s)'%(nice_name, value, name))
    if not min is None :
        if v < min :
            raise InputConversionError('%s=%f less than minimum allowed value %f (%s)'%(nice_name, v, min, name))
    if not max is None :
        if v > max :
            raise InputConversionError('%s=%f greater than maximum allowed value %f (%s)'%(nice_name, v, max, name))
    return v

# WARNING:  overriding builtin functions min() and max() as parameters
def int_range( value, name, min=None, max=None ) :
    #get a nice name for the errored key
    nice_name=lookup_nice_info(name)
    try :
        v = int(value)
    except:
        raise InputConversionError('%s (%s) is not an integer (%s)'%(nice_name, value, name))
    if not min is None :
        if v < min :
            raise InputConversionError('%s=%d less than minimum allowed value %d (%s)'%(nice_name, v, min, name))
    if not max is None :
        if v > max :
            raise InputConversionError('%s=%d greater than maximum allowed value %d (%s)'%(nice_name, v, max, name))
    return v

#
def list_int( value, name, l ) :
    try :
        v = int(value)
    except:
        raise InputConversionError('%s is not an integer (%s)'%(lookup_nice_info(name),name))
    if not v in l:
        raise InputConversionError('%s not one of the choices in the list for %s (%s)'%(value,lookup_nice_info(name), name))
    return v

#
def list_str( value, name, valid ) :
    if value in valid :
        # str() to de-unicode it
        return str(value)
    raise InputConversionError('%s not one of the choices in the list for %s'
                     %(value,lookup_nice_info(name)))


# these functions deal with R.A. and Dec. validation
_format_spec = '%s not a valid equatorial coordinate'

def _sex2dec(value):
    try:
        v,m,s = value.split(":")
        v = float(v)
        if v != 0.0:
            signal = v / abs(v)
        else:
            signal = 1.

        result = signal * (abs(v) + float(m)/60. + float(s)/3600.)
        return result

    except ValueError:
        raise InputConversionError(_format_spec % value)

def _check_range(value, value_to_check, min, max):
    # 'value-to_check' is the actual float value to range-check.
    # 'value' is the original input string, used for reporting only.
    if min and value_to_check < min:
        raise InputConversionError(_format_spec % value)
    if max and value_to_check > max:
        raise InputConversionError(_format_spec % value)

#

# A declination looks like
# [+-]00:00:00.00000
#
# degress:minutes:seconds.frac
#
# but etc_engine apparently 1) wants a string, and 2) only allows
# integer seconds
#
re_dec = re.compile('^ *[-+]?[0-9]+:[0-9]+:[0-9]+ *$')

def declination( value, name, min=None, max=None ) :
    # str() to de-unicode it
    value = str(value)
    try:
        # declination in decimal degrees
        value_to_check = float(value)
        _check_range(value, value_to_check, min, max)
        return value

    except ValueError:
        pass

    if not re_dec.match(value) :
        raise InputConversionError(_format_spec % value)

    # declination in sexagesimal degrees
    value_to_check = _sex2dec(value)
    _check_range(value, value_to_check, min, max)
    return value

# A Right Ascension looks like
# 00:00:00.00000
#
# hours:minutes:seconds.frac
#
# but etc_engine apparently 1) wants a string, and 2) only allows
# integer seconds
#
re_ra_1 = re.compile('^ *[0-9]+:[0-9]+:[0-9]+ *$')
# re_ra_1 = re.compile('[0-9]+:[0-9]+:[0-9]+[.]*[0-9]*')
def right_ascension( value, name, min=None, max=None ) :
    # str() to de-unicode it
    value = str(value)
    try:
        # R.A. in decimal degrees
        value_to_check = float(value)
        _check_range(value, value_to_check, min, max)
        return value

    except ValueError:
        pass

    if not re_ra_1.match(value) :
        raise InputConversionError(_format_spec % value)

    # R.A. in hh:mm:ss format
    value_to_check = _sex2dec(value)
    value_to_check *= 360./24.
    _check_range(value, value_to_check, min, max)
    return value


###
#
re_rectangle = re.compile('([0-9]+)|([0-9]+,[0-9]+)')

def rectangle( value, name ) :
    if not re_rectangle :
        raise InputConversionError('%s not a valid rectangle'%name)
    return str(value)

#
default_check = alnum

# This dict contains a list of all valid field names (for all forms)
# and the validation/conversion function to use for each.
#
# Each is on a single very wide line because it makes it easy to keep
# them sorted.  In vi, move to the first line and type "!}./icf" to
# pipe it through the script icf to sort/format the list.  emacs has
# a way to do this too, but I don't remember what it is.  You have
# to use some variant of shell-command-on-region.
#
# It isn't really necessary to give detailed lists here in most cases:
#  - If a value came from a drop-list on the web form, we know it is
#    acceptable for any user who is not trying to circumvent our system. If
#    they _are_, we don't care if they get the wrong answer.
#  - If a value is only ever used as a key to look things up in a dict,
#    it is sufficient to use something simple like alnum.  Values not in the
#    dict will raise an exception anyway, and we don't need to maintain
#    the lists here.
#  - If a value is a number, the engine can't be hurt by it.
#
# Remember the checks here are to guard against being hurt by user input.
# Questions of whether the input values _make_ _sense_ do not belong here.
#

# NOTE - if you add or delete items from here, please consider whether
#        you should also edit anything in etc_web/bit/config ...

converter = {

    'AirglowStandard'                   : alnum,
    'E140H_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1271, 1234, 1307, 1343, 1380, 1416, 1453, 1489, 1526, 1562, 1598]),
    'E140M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1425]),
    'E230H_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1763, 1813, 1863, 1913, 1963, 2013, 2063, 2113, 2163, 2213, 2263, 2313, 2363, 2413, 2463, 2513, 2563, 2613, 2663, 2713, 2762, 2812, 2862, 2912, 2962, 3012]),
    'E230M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1978, 2124, 2269, 2415, 2561, 2707]),
    'EarthshineMag'                     : float_range,
    'EarthshineMult'                    : lambda v,n : float_range(v, n, min=0),
    'EarthshineSpec'                    : alnum,
    'EarthshineStandard'                : alnum,
    'G130M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1055, 1096, 1222, 1291, 1300, 1309, 1318, 1327, ] ),
    'G140L_CentralWavelength'           : lambda v,n : list_int( v,n, [ 800, 1105, 1280, ]),
    'G140M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1173, 1218, 1222, 1272, 1321, 1371, 1387, 1400, 1420, 1470, 1518, 1540, 1550, 1567, 1616, 1640, 1665, 1714]),
    'G160M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1533, 1577, 1589, 1600, 1611, 1623, ]),
    'G185M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1786, 1817, 1835, 1850, 1864, 1882, 1890, 1900, 1913, 1921, 1941, 1953, 1971, 1986, 2010, ]),
    'G225M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 2186, 2217, 2233, 2250, 2268, 2283, 2306, 2325, 2339, 2357, 2373, 2390, 2410, ] ),
    'G230L_CentralWavelength'           : lambda v,n : list_int( v,n, [ 2635, 2950, 3000, 3360, ] ),
    'G230MB_CentralWavelength'          : lambda v,n : list_int( v,n, [ 1713, 1854, 1995, 2135, 2276, 2416, 2557, 2697, 2794, 2836, 2976, 3115] ),
    'G230M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 1687, 1769, 1851, 1884, 1933, 2014, 2095, 2176, 2257, 2338, 2419, 2499, 2579, 2600, 2659, 2739, 2800, 2818, 2828, 2898, 2977, 3055]),
    'G285M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 2617, 2637, 2657, 2676, 2695, 2709, 2719, 2739, 2850, 2952, 2979, 2996, 3018, 3035, 3057, 3074, 3094, ]),
    'G430M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 3165, 3305, 3423, 3680, 3843, 3936, 4194, 4451, 4706, 4781, 4961, 5093, 5216, 5471] ),
    'G750M_CentralWavelength'           : lambda v,n : list_int( v,n, [ 5734, 6094, 6252, 6581, 6768, 7283, 7795, 8311, 8561, 8825, 9286, 9336, 9806, 9851] ),
    'HeliumStandard'                    : lambda v,n : list_str(v,n, ['None','Average','High','Very High']),
    'Mirror'                            : lambda v,n : list_str(v,n, ['mirrora', 'mirrorb' ]),
    'SNR'                               : float_range,
    'Stripe'                            : lambda v,n : list_str(v, n, ['A', 'B', 'C'] ),
    'Time'                              : lambda v,n : float_range(v, n, min=0),
    'scRate'                            : lambda v,n : float_range(v, n, min=0),
    'scLengthFromRate'                  : lambda v,n : float_range(v, n, min=0),
    'scLengthFromTime'                  : lambda v,n : float_range(v, n, min=0),
    'ZodiDate'                          : lambda v,n : int_range(v,n, min=1990),
    'ZodiDay'                           : lambda v, n: int_range(v, n, min=1, max=31),
    'ZodiDec'                           : lambda v, n: declination(v, n, min=-90.0, max=90.0),
    'ZodiMag'                           : float_range,
    'ZodiMonth'                         : lambda v, n: int_range(v, n, min=1, max=12),
    'ZodiMult'                          : lambda v,n : float_range(v, n, min=0),
    'ZodiRA'                            : lambda v, n: right_ascension(v, n, min=0.0, max=360.0),
    'ZodiSpec'                          : alnum,
    'ZodiStandard'                      : alnum,
    'ZodiSun'                           : lambda v,n : float_range(v, n, min=0.0, max=360.0),
    'ZodiSunAttribute'                  : alnum,
    'binx'                              : lambda v,n : list_int(v, n, [ 1,2,4 ]),
    'biny'                              : lambda v,n : list_int(v, n, [ 1,2,4 ]),
    'ccdMode'                           : lambda v,n : list_str(v, n, [ 'ACQ', 'ACQ/PEAK' ]),
    'ccdaperture0'                      : lambda v,n : list_str(v, n, ['0.2X0.2','52X0.05','52X0.1','52X0.2','52X0.5','52X2','50CCD','F28X50LP','F25ND3','F25ND5','F28X50OII','F28X50OIII','F25NDQ1','F25NDQ2','F25NDQ3','F25NDQ4','0.2X0.05ND','0.3X0.05ND','0.1X0.03','0.2X0.06','0.2X0.09','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'ccdapertureACQ'                    : al_dot_num,
    'ccdapertureACQPEAK'                : al_dot_num,
    'ccddarklevel'                      : lambda v,n : list_str(v, n, [ 'high', 'medium', 'low' ] ),
    'cosaperture0'                      : lambda v,n : list_str(v, n, [ 'PSA', 'BOA' ] ),
    'crsplit'                           : lambda v, n: int_range(v, n, min=1),
    'detector'                          : alnum,
    'disperser'                         : alnum,
    'e140haperture0'                    : lambda v,n : list_str(v, n, ['0.2X0.09','0.2X0.2','6X0.2','0.1X0.03','0.2X0.05ND','0.3X0.05ND', '52X0.05','25MAMA', 'F25QTZ','F25SRF2', 'F25ND3', 'F25ND5','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'e140maperture0'                    : lambda v,n : list_str(v, n, ['0.2X0.06','0.2X0.2','6X0.2','0.1X0.03','0.2X0.05ND','0.3X0.05ND', '52X0.05','25MAMA', 'F25QTZ','F25SRF2', 'F25ND3', 'F25ND5','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'e230haperture0'                    : lambda v,n : list_str(v, n, ['0.2X0.09','0.2X0.2','6X0.2','0.1X0.03','0.1X0.09','0.1X0.2','0.2X0.05ND','0.3X0.05ND', '52X0.05','25MAMA', 'F25QTZ','F25SRF2', 'F25MGII', 'F25ND3', 'F25ND5','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'e230maperture0'                    : lambda v,n : list_str(v, n, ['0.2X0.06','0.2X0.2','6X0.2','0.1X0.03','0.2X0.05ND','0.3X0.05ND', '52X0.05','25MAMA', 'F25QTZ','F25SRF2', 'F25MGII', 'F25ND3', 'F25ND5','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'extractionRegionCircle'            : lambda v,n : float( list_str( v, n, ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1.0", "1.5", "2.0", 0.4,] )),
    'extractionRegionExtendedCircle'    : lambda v,n : float( list_str( v, n, ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1.0", "1.5", "2.0", 0.4,] )),
    'extractionRegionExtendedSquare'    : lambda v,n : int_range( v, n, 0, 200 ),
    'extractionRegionExtendedUserCircle': lambda v,n : float_range(v, n, min=0),
    'extractionRegionExtendedUserCircle_min': float_range,
    'extractionRegionExtendedUserCircle_max': float_range,
    'extractionRegionRectangle'         : rectangle,
    'extractionRegionScHeight'          : alnum,
    'extractionRegionScWidth'           : alnum,
    'extractionRegionSquare'            : lambda v,n : list_int( v, n, [ 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 51, 101, ] ),
    'extractionRegionUserCircle'        : lambda v,n : float_range(v, n, min=0),
    'extractionRegionUserCircle_min'    : float_range,
    'extractionRegionUserCircle_max'    : float_range,
    'extractionRegionVariableCircle'    : lambda v,n : float_range(v, n, min=0, max=100.),
    'fband'                             : alnum,
    'fbandpass'                         : alnum,
    'fIndex'                            : float_range,
    'fIsLambda'                         : lambda v,n : list_str(v,n,['true', 'false'] ),
    'fL1_center'                        : lambda v,n : float_range(v, n, min=0),
    'fL1_flux'                          : lambda v,n : float_range(v, n, min=0),
    'fL1_fwhm'                          : lambda v,n : float_range(v, n, min=0),
    'fL2_center'                        : lambda v,n : float_range(v, n, min=0),
    'fL2_flux'                          : lambda v,n : float_range(v, n, min=0),
    'fL2_fwhm'                          : lambda v,n : float_range(v, n, min=0),
    'fL3_center'                        : lambda v,n : float_range(v, n, min=0),
    'fL3_flux'                          : lambda v,n : float_range(v, n, min=0),
    'fL3_fwhm'                          : lambda v,n : float_range(v, n, min=0),
    'fOtherFile'                        : lambda v,n : re_check(v,n, '^[$A-Za-z0-9._]*$'), # file names in cdbs in IRAF format
    'fRedshift'                         : lambda v,n : float_range(v, n, min=0),
    'fSpectrumCK'                       : alnum,
    'fSpectrumPickles'                  : alnum,
    'fStellar'                          : al_dot_num,
    'fUploadFile'                       : upload,
    'fbbtemp'                           : lambda v,n : float_range(v, n, min=0),
    'fbpgsfile'                         : alnum,
    'fcalfile'                          : lambda v,n : re_check(v,n, '^[A-Za-z0-9._+-]*$'),
    'fphoenixfile'                      : lambda v,n : re_check(v,n, '^[A-Za-z0-9._+-]*$'),
    'fdiameter'                         : float_range,
    'febmvtype'                         : lambda v,n : list_str(v,n,[ "mwavg", "mwdense", "mwrv21","mwrv4","lmcavg", "lmc30dor", "smcbar", "xgalsb"]),
    'febv'                              : lambda v,n : float_range(v, n, min=0),
    'fextinctiontype'                   : lambda v,n : list_str(v,n,[ "before", "after" ]),
    'rn_flux_lambda'                    : float_range,
    'rn_flux_lambda_units'              : lambda v,n : list_str(v, n,['flam','jy','abmag']),
    'rn_flux_bandpass'                    : float_range,
    'rn_flux_bandpass_units'              : lambda v,n : list_str(v, n,['flam','jy','abmag', 'vegamag']),
    'rn_lambda'                         : float_range,
    'rn_lambda_units'                   : lambda v,n : list_str(v, n,['angstroms','nanometers','microns']),
    'fftype'                            : al_dot_num,
    'fftype_filters'                    : al_dot_num,
    'fnonstellar'                       : lambda v,n : list_str(v,n, [
        'Gliese 229B',
        'Gliese 752B',
        'Gliese 411',
        'Gliese 406',
        'Orion',
        'Orion2',
        'PN',
        'PN Extended',
        'QSO',
        'QSO2',
        'QSO SDSS',
        'QSO IRTF',
        'QSO COS ',
        'Elliptical',
        'Elliptical2',
        'El B2004a' ,
        'El CWW FUV',
        'NGC1068',
        'Im B2004a',
        'Spiral',
        'S0 FUV CWW',
        'Sa FUV CWW',
        'Sbc B2004a',
        'Sbc CWW',
        'Scd B2004a',
        'SSP 25myr z008',
        'SSP 5myr z008',
        'Tau06 z02 1015 190',
        'Tau06 z02 2500 000',
        'Tau06 z02 4500 000',
        'Tau06 z02 5000 000',
        'Tau06 z02 6000 000',
        'Tau06 z02 8000 000',
        'Tau06 z02 12000 000',
        'Sb1 Kinney FUV',
        'Sb2 B2004a',
        'Sb2 Kinney FUV',
        'Sb3 B2004a',
        'Sb3 Kinney FUV',
        'Sb4 Kinney FUV',
        'Sb5 Kinney FUV',
        'Sb6 Kinney FUV',
        'ULIRG05189-2524',
        'ULIRG12112+0305',
        'ULIRG14348+1447',
        'ULIRG15250+3609',
        'JWIRAS22491-1808',
        'ARP220',
        'SB_M82',
        'MRK_1014',
        'MRK_231',
        'MRK_273',
        'MRK_463',
        'NGC6240',
        'UGC5101'
        ]),
    'fsorigin'                          : lambda v,n : list_str(v, n, [ 'SpectrumUpload', 'SpectrumHstOther', 'SpectrumCK', 'SpectrumPickles', 'SpectrumKurucz', 'SpectrumStellar', 'SpectrumHST', 'SpectrumPhoenix', 'SpectrumNonStellar', 'SpectrumBlackBody', 'SpectrumPowerLaw', 'SpectrumFlat', 'SpectrumEmpty', ] ),
    'fsourceType'                       : lambda v,n : list_str(v, n, [ 'extended', 'point' ] ),
    'filter.ubvri'                      : lambda v,n : list_str(v,n, ["Johnson/U", "Johnson/B", "Johnson/V", "Johnson/R", "Johnson/I", "Johnson/J", "Johnson/K", "Bessell/H", "Bessell/K", "Bessell/J","Cousins/R", "Cousins/I"]),
    'filter.sloan'                      : lambda v,n : list_str(v,n, ["Galex/FUV", "Galex/NUV", "Sloan/U", "Sloan/G", "Sloan/R", "Sloan/I", "Sloan/Z"]),
    'filter.nicmos'                     : lambda v,n : list_str(v,n, ["NICMOS/F110W", "NICMOS/F160W"]),
    'filter.acs'                        : lambda v,n : list_str(v,n, ["ACS/F435W", "ACS/F475W", "ACS/F555W", "ACS/F606W", "ACS/F625W", "ACS/F775W", "ACS/F814W", "ACS/F850LP"]),
    'filter.wfc3UVIS'                   : lambda v,n : list_str(v,n, ["WFC3/UVIS/F218W", "WFC3/UVIS/F200LP", "WFC3/UVIS/F225W", "WFC3/UVIS/F275W", "WFC3/UVIS/F300X", "WFC3/UVIS/F336W", "WFC3/UVIS/F350LP", "WFC3/UVIS/F390W", "WFC3/UVIS/F438W", "WFC3/UVIS/F475X","WFC3/UVIS/F600LP"]),
    'filter.wfc3IR'                     : lambda v,n : list_str(v,n, ["WFC3/IR/F098M", "WFC3/IR/F105W", "WFC3/IR/F110W", "WFC3/IR/F125W", "WFC3/IR/F140W", "WFC3/IR/F160W"]),
    'fuvmamaaperture0'                  : lambda v,n : list_str(v, n, ['0.2X0.2','52X0.05','52X0.1','52X0.2','52X0.5','52X2','25MAMA','F25NDQ1','F25NDQ2','F25NDQ3','F25NDQ4','F25ND3','F25ND5','F25QTZ','F25SRF2','F25LYA','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'gain'                              : float_range,
    'hrcfilt0'                          : lambda v,n : alnum(v,n),
    'hrcfilt1'                          : lambda v,n : alnum(v,n),
    'instrument'                        : alnum,
    'irfilt0'                           : alnum,
    'cosAcqMode'                        : lambda v,n : list_str(v, n, [ 'image', 'search' ] ),
    'fuvglowregion'                     : lambda v,n : list_str(v, n, [ 'high', 'medium', 'low', 'none' ] ),
    'nuvMode'                           : lambda v,n : list_str(v, n, ['ACQ', 'ACQ/PEAKXD','ACCUM'] ),
    'nuvmamaaperture0'                  : lambda v,n : list_str(v, n, ['0.2X0.2','52X0.05','52X0.1','52X0.2','52X0.5','52X2','25MAMA','F25NDQ1','F25NDQ2','F25NDQ3','F25NDQ4','F25ND3','F25ND5','F25QTZ','F25SRF2', 'F25MGII', 'F25CN270','F25CN182', 'F25CIII','31X0.05NDA','31X0.05NDB','31X0.05NDC']),
    'obswave'                           : lambda v,n : float_range(v, n, min=0, max=20000),
    'post_flash_acs'                    : lambda v, n: int_range(v, n, min=0, max=5733),
    'post_flash_wfc3'                   : lambda v, n: int_range(v, n, min=0, max=25),
    'prismaperture0'                    : lambda v,n : list_str(v, n, ['52X0.05', '52X0.1','52X0.2','52X0.5','52X2','F25QTZ','F25SRF2','F25MGII','F25NDQ1','F25NDQ2','F25NDQ3','F25NDQ4','F25ND3','F25ND5','25MAMA']),
    'sbcfilt0'                          : lambda v,n : list_str(v, n, [ 'F122M','F115LP','F125LP','F140LP','F150LP','F165LP'] ),
    'science_mode'                      : alnum,
    'simmode'                           : lambda v,n : list_str(v, n, [ 'Time', 'SNR', 'SNR_from_rate' ]),
    'wavelengthUnits'                   : alnum,
    'wfc3_filter_m'                     : alnum,
    'wfc3_filter_n'                     : alnum,
    'wfc3_filter_q'                     : alnum,
    'wfc3_filter_type'                  : lambda v,n : list_str(v, n, [ 'medium','wide','narrow','quad']),
    'wfc3_filter_w'                     : alnum,
    'wfcfilt0'                          : lambda v,n : alnum(v,n),
    'wfcfilt1'                          : lambda v,n : alnum(v,n),
    'xRegionExtendedType'               : lambda v,n : list_str(v, n, ['default','Default', 'Circle', 'Square', 'Rectangle', 'UserCircle'] ),
    'xRegionType'                       : lambda v,n : list_str(v, n, [ 'default', 'Default', 'Square', 'Circle', 'UserCircle', 'VariableCircle', 'Rectangle', 'scRectangle' ] ),
}

#
# All of this is a consistency check that happens in developer systems.
# - are there fields in the form that are not listed in converter[] ?
# - are there fields in converter[] that are not listed in any form ?
#

if CFG_DEBUG :
    check_fields_missing = { }
    extra_fields = converter.copy()

    def check_fields(b) :
        for x in b.field_list :
            if not x in converter :
                if x in check_fields_missing:
                    check_fields_missing[x].append( b.__file__ )
                else :
                    check_fields_missing[x] = [ b.__file__ ]
            if x in extra_fields :
                del extra_fields[x]

    import pandeia.engine.helpers.bit.instruments.hst.acs.web.form_fields_acs_imaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.acs.web.form_fields_acs_rampfilter as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.acs.web.form_fields_acs_spectroscopic as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.cos.web.form_fields_cos_imaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.cos.web.form_fields_cos_spectroscopic as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.cos.web.form_fields_cos_spectroscopicacq as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.cos.web.form_fields_cos_targetacquisition as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.stis.web.form_fields_stis_imaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.stis.web.form_fields_stis_spectroscopic as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.stis.web.form_fields_stis_targetacquisition as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3ir.web.form_fields_wfc3ir_imaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3ir.web.form_fields_wfc3ir_scimaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3ir.web.form_fields_wfc3ir_spectroscopic as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3ir.web.form_fields_wfc3ir_scspectroscopic as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3uvis.web.form_fields_wfc3uvis_imaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3uvis.web.form_fields_wfc3uvis_scimaging as a
    check_fields(a)
    import pandeia.engine.helpers.bit.instruments.hst.wfc3uvis.web.form_fields_wfc3uvis_spectroscopic as a
    check_fields(a)

    if len(check_fields_missing) > 0 :
        sys.stderr.write(
            "\n\nMISSING FIELDS (required by some form) IN INPUT VALIDATOR: %s\n"
            %__file__
        )
        for X in check_fields_missing :
            sys.stderr.write( "    %s\n" % X )
            for y in check_fields_missing[X] :
                sys.stderr.write("            %s\n" % os.path.basename(y))
        sys.stderr.write('\n')
        sys.stderr.flush()

    if len(extra_fields) > 0 :
        sys.stderr.write(
            "\n\nEXTRA FIELDS (not expected by any form) IN INPUT VALIDATOR: %s\n"
            % __file__
        )
        for X in extra_fields :
            sys.stderr.write("    %s\n" % X)
        sys.stderr.write("\n")
        sys.stderr.flush()

# =========================================================================

PLOT_TYPES = [
    "Observed Target Spectrum",
    "Input Target Spectrum",
    "Signal-to-noise",
    "Throughput",
    "Total Counts",
    "Re-plot"
    ]

LEGEND_LOCATIONS = [
    "best",
    "upper right",
    "center right",
    "lower right",
    "upper left",
    "center left",
    "lower left",
    "upper center",
    "center",
    "lower center",
]

INT_RE = re.compile(r"\d+")
def chart_pixels(pstr):
    """Convert to integer and validate a chart X or Y pixel dimension."""
    if INT_RE.match(pstr):
        pix = int(pstr)
        if 350 <= pix <= 3000:
            return pix
    raise InputConversionError("size must be a decimal number in the range 350..3000")

FLOAT_RE = re.compile(r"^([+/-]?((([0-9]+(\.)?)|([0-9]*\.[0-9]+))([eE][+\-]?[0-9]+)?))$")
def valid_float(fstr):
    """Convert suitable strings into floats,  exclude expressions."""
    if FLOAT_RE.match(fstr):
        return float(fstr)
    raise InputConversionError("badly formatted real number " + repr(str(fstr)))

#
# These are all the inputs that may be given to a plot request, along
# with the validator to use for it.
#
PLOT_INPUTS =     {   # Map form variables to validator kinds
        "LogX" : "checkbox",
        "LogY" : "checkbox",
        "ChartXMin" : valid_float,
        "ChartXMax" : valid_float,
        "ChartYMin" : valid_float,
        "ChartYMax" : valid_float,
        "ChartXSize" : chart_pixels,
        "ChartYSize" : chart_pixels,
        "FullRange" : "checkbox",
        "PlotType" : PLOT_TYPES,
        "PlotTypeVal": PLOT_TYPES,
        "LegendLocation":LEGEND_LOCATIONS,
      }

PLOT_ERR_MAP = {   # Map form variables to plot page oriented text
        "ChartXMin" : "Axis Min, Wavelength",
        "ChartYMin" : "Axis Min, Y",
        "ChartXMax" : "Axis Max, Wavelength",
        "ChartYMax" : "Axis Max, Y",
        "ChartXSize" : "Plot Size, Wavelength",
        "ChartYSize" : "Plot Size, Y",
        "LegendLocation" : "Legend Location",
    }

def plot_input_conversion(request):
    """Fetches and validates plot form POST variables from `request`
    returning clean form variable dictionary `fdict`.
    """
    # We collect all the valid inputs in fdict and then return it.
    # Invalid fields are just ignored because the are not listed in PLOT_INPUTS.
    fdict = {}
    for var, validator in list(PLOT_INPUTS.items()):
        if var in request.POST:
            value = request.POST[var]
            if value == "":
                continue
            if isinstance(validator, list):
                if value in validator:
                    fdict[var] = value
                else:
                    raise InputConversionError("Unexpected value for " + repr(PLOT_ERR_MAP[var]) + " not one of: " + str(validator))
            elif validator == "checkbox":
                fdict[var] = "checked"
            else:
                try:
                    fdict[var] = validator(value)
                except InputConversionError as e:
                    raise InputConversionError("Invalid " + repr(PLOT_ERR_MAP[var]) + " : " + str(e))
                except:
                    raise InputConversionError("Invalid input for " + repr(PLOT_ERR_MAP[var]) + " = " + repr(str(value)))
    return fdict
