:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :private-members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: /modules/generated/backreferences/{{module}}.{{objname}}.examples
