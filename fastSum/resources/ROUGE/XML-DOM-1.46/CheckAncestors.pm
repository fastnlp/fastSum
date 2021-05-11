#
# Perl module for testing the XML::DOM module.
# Used by the test cases in the 't' directory.
# Recursively walks the node tree and checks parent/child and document links.
#

use strict;

package CheckAncestors;

use XML::DOM;
use Carp;

BEGIN
{
    # import the constants for accessing member fields, e.g. _Doc
    import XML::DOM::Node qw{ :Fields };
    import XML::DOM::DocumentType qw{ :Fields };
}

sub new
{
    my %args = (Mark => {});
    bless \%args, $_[0];
}

sub check
{
    my ($self, $node) = @_;

    # check if node was already seen
    croak "found Node twice [$node]" if ($self->{Mark}->{$node});
    $self->{Mark}->{$node} = $node;

    # check if document is correct
    my $doc = $self->{Doc};
    if (defined $doc)
    {
	my $doc2 = $node->[_Doc];
	croak "wrong Doc [$doc] [$doc2]" if $doc != $doc2;
    }
    else
    {
	$self->{Doc} = $doc;
    }
    
    # check if node's children know their parent
    # and, recursively, check each kid
    my $nodes = $node->getChildNodes;
    if ($nodes)
    {
	for my $kid (@$nodes)
	{
	    my $parent = $kid->getParentNode;
	    croak "wrong parent node=[$node] parent=[$parent]"
		if ($parent != $node);
	    $self->check ($kid);
	}
    }

    # check NamedNodeMaps
    my $type = $node->getNodeType;
    if ($type == XML::DOM::Node::ELEMENT_NODE || 
	$type == XML::DOM::Node::ATTLIST_DECL_NODE)
    {
	$self->checkAttr ($node, $node->[_A]);
    }
    elsif ($type == XML::DOM::Node::DOCUMENT_TYPE_NODE)
    {
	$self->checkAttr ($node, $node->[_Entities]);
	$self->checkAttr ($node, $node->[_Notations]);	
    }
}

# (This should have been called checkNamedNodeMap)
sub checkAttr
{
    my ($self, $node, $attr) = @_;
    return unless defined $attr;

    # check if NamedNodeMap was already seen
    croak "found NamedNodeMap twice [$attr]" if ($self->{Mark}->{$attr});
    $self->{Mark}->{$attr} = $attr;

    # check if document is correct
    my $doc = $self->{Doc};
    if (defined $doc)
    {
	my $doc2 = $attr->getProperty ("Doc");
	croak "wrong Doc [$doc] [$doc2]" if $doc != $doc2;
    }
    else
    {
	$self->{Doc} = $attr->getProperty ("Doc");
    }

    # check if NamedNodeMap knows his daddy
    my $parent = $attr->getProperty ("Parent");
    croak "wrong parent node=[$node] parent=[$parent]"
	unless $node == $parent;

    # check if NamedNodeMap's children know their parent
    # and, recursively, check the child nodes
    my $nodes = $attr->getValues;
    if ($nodes)
    {
	for my $kid (@$nodes)
	{
	    my $parent = $kid->{InUse};
	    croak "wrong InUse attr=[$attr] parent=[$parent]"
		if ($parent != $attr);

	    $self->check ($kid);
	}
    }
}

sub doit
{
    my $node = shift;
    my $check = new CheckAncestors;
    eval {
	$check->check ($node);
    };
    if ($@)
    {
	print "checkAncestors failed:\n$@\n";
	return 0;
    }
    return 1;
}

1;
