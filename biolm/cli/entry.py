#!/usr/bin/env python
"""Entry point wrapper to suppress urllib3 warnings."""
import sys
import os
import warnings

# CRITICAL: Redirect stderr FIRST before anything else can print to it
_devnull = open(os.devnull, 'w')
_original_stderr_fd = os.dup(2)  # Save original stderr file descriptor
os.dup2(_devnull.fileno(), 2)  # Redirect stderr (fd 2) to /dev/null
sys.stderr = _devnull

# Suppress ALL warnings aggressively
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Specifically suppress urllib3/OpenSSL warnings
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='.*LibreSSL.*')
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*')

# Set environment variable before any imports
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Monkey-patch warnings.warn before urllib3 imports to catch it at the source
_original_warn = warnings.warn
def _silent_warn(message, category=None, stacklevel=1, source=None):
    msg_str = str(message) if message else ''
    if any(keyword in msg_str.lower() for keyword in ['urllib3', 'openssl', 'libressl', 'notopenssl']):
        return  # Silently ignore urllib3 warnings
    _original_warn(message, category, stacklevel, source)
warnings.warn = _silent_warn

# Try to import urllib3 early and disable ALL its warnings
# This must happen while stderr is redirected
try:
    import urllib3
    # Disable all urllib3 warnings
    urllib3.disable_warnings()
    # Specifically disable NotOpenSSLWarning if it exists
    try:
        urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
    except (AttributeError, TypeError):
        pass
    # Also try to disable at the logging level
    try:
        import logging
        logging.getLogger('urllib3').setLevel(logging.ERROR)
    except:
        pass
except (ImportError, AttributeError):
    pass

# Patch showwarning to aggressively filter urllib3/OpenSSL warnings
_original_showwarning = warnings.showwarning
def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    msg_str = str(message) if message else ''
    filename_str = str(filename) if filename else ''
    category_str = str(category) if category else ''
    
    # Filter out any urllib3, OpenSSL, LibreSSL, or NotOpenSSLWarning related warnings
    if any(keyword in msg_str.lower() or keyword in filename_str.lower() or keyword in category_str.lower() 
           for keyword in ['urllib3', 'openssl', 'libressl', 'notopenssl']):
        return
    _original_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _filtered_showwarning

# Import CLI (urllib3 will be imported via httpx)
from biolm.cli import cli

# Create a filtered stderr that suppresses urllib3 warnings
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self._buffer = ''
    
    def write(self, text):
        # Handle both str and bytes
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        # Buffer text to handle multi-line warnings
        if text:
            self._buffer += text
            # Check if buffer contains urllib3 warning (handle both single and multi-line)
            buffer_lower = self._buffer.lower()
            if any(keyword in buffer_lower 
                   for keyword in ['urllib3', 'openssl', 'libressl', 'notopensslwarning', 
                                   'urllib3 v2 only supports', 'libressl 2.8.3']):
                # Clear buffer and don't write
                self._buffer = ''
                return
            # If we have a complete line (ends with newline), write it and clear buffer
            if '\n' in text:
                self.original_stderr.write(self._buffer)
                self._buffer = ''
    
    def flush(self):
        if self._buffer:
            # Check buffer one more time before flushing
            if not any(keyword in self._buffer.lower() 
                      for keyword in ['urllib3', 'openssl', 'libressl', 'notopensslwarning']):
                self.original_stderr.write(self._buffer)
            self._buffer = ''
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Restore stderr but wrap it with filtering
os.dup2(_original_stderr_fd, 2)  # Restore original stderr file descriptor
sys.stderr = FilteredStderr(sys.__stderr__)
_devnull.close()
os.close(_original_stderr_fd)

if __name__ == '__main__':
    sys.exit(cli())
