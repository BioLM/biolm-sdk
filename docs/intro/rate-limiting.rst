========================
Rate Limiting and Throttling
========================

The client batches requests and runs them concurrently. By default it uses the API’s recommended throttle and a concurrency limit (up to 16 in-flight requests). You can leave defaults, set a custom rate (e.g. requests per second or minute), or set a custom concurrency limit.

**Examples:**

.. code-block:: python

    from biolm.core.http import BioLMApi, BioLMApiClient

    # Use API's default throttle rate (recommended)
    model = BioLMApi("esmfold")

    # Custom rate limit: 1000 requests per second
    model = BioLMApi("esmfold", rate_limit="1000/second")

    # Custom rate limit: 60 requests per minute
    model = BioLMApi("esmfold", rate_limit="60/minute")

    # Custom concurrency limit: at most 5 requests at once
    model = BioLMApiClient("esmfold", semaphore=5)
    
    # Or use your own semaphore
    import asyncio
    sem = asyncio.Semaphore(5)
    model = BioLMApiClient("esmfold", semaphore=sem)
    
    # Disable default semaphore (no concurrency limit)
    model = BioLMApiClient("esmfold", semaphore=None)

    # Both: at most 5 concurrent, and at most 1000 per second
    model = BioLMApiClient("esmfold", semaphore=sem, rate_limit="1000/second")

------------------------
Implementation Details
------------------------

- **Semaphore**: Limits the number of requests in flight at any moment.
- **Rate Limiter**: Tracks timestamps of recent requests and enforces the N-per-window rule.
- **Sliding Window**: The limiter removes timestamps older than the window (1s or 60s) and only allows a new request if the count is below N.
- **Acquisition Order**: Semaphore is acquired first, then the rate limiter.

------------------------
Best Practices
------------------------

- For most users, the default API throttle is sufficient and safest.
- Use a custom rate limit if you have a dedicated quota or want to avoid 429 errors.
- Use a semaphore to avoid overwhelming your own network or compute resources.
- For very high throughput, combine both.

------------------------
See Also
------------------------

- :doc:`error-handling`
- :doc:`batching`
- :doc:`../faq`
