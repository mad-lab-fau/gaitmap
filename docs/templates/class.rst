:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: {{module}}.{{objname}}.examples
