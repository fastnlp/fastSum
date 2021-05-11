#
# Used by test scripts to compare 2 DOM subtrees.
# 
# Usage:
#
# my $cmp = new CmpDOM;
# $node1->equals ($node2, $cmp) or 
#   print "Difference found! Context:" . $cmp->context . "\n";
#
use strict;

package CmpDOM;

use XML::DOM;
use Carp;

sub new
{
    my %args = (SkipReadOnly => 0, Context => []);
    bless \%args, $_[0];
}

sub pushContext
{
    my ($self, $str) = @_;
    push @{$self->{Context}}, $str;
#print ":: " . $self->context . "\n";
}

sub popContext
{
    pop @{$_[0]->{Context}};
}

sub skipReadOnly
{
    my $self = shift;
    my $prev = $self->{SkipReadOnly};
    if (@_ > 0)
    {
	$self->{SkipReadOnly} = shift;
    }
    $prev;
}

sub sameType
{
    my ($self, $x, $y) = @_;

    return 1 if (ref ($x) eq ref ($y));

    $self->fail ("wrong type " . ref($x) . " != " . ref($y));
}

sub sameReadOnly
{
    my ($self, $x, $y) = @_;
    return 1 if $self->{SkipReadOnly};

    my $result = 1;
    if (not defined $x)
    {
	$result = 0 if defined $y;
    }
    else
    {
	if (not defined $y)
	{
	    $result = 0;
	}
	elsif ($x != $y)
	{
	    $result = 0;
	}
    }
    return 1 if ($result == 1);

    $self->fail ("ReadOnly $x != $y");
}

sub fail
{
    my ($self, $str) = @_;
    $self->pushContext ($str);
    0;
}

sub context
{
    my $self = shift;
    join (", ", @{$self->{Context}});
}

package XML::DOM::NamedNodeMap;

sub equals
{
    my ($self, $other, $cmp) = @_;

    return 0 unless $cmp->sameType ($self, $other);

    # sanity checks
    my $n1 = int (keys %$self);
    my $n2 = int (keys %$other);
    return $cmp->fail("same keys length") unless $n1 == $n2;

    return $cmp->fail("#1 value length") unless ($n1-1 == $self->getLength);
    return $cmp->fail("#2 value length") unless ($n2-1 == $other->getLength);

    my $i = 0;
    my $ov = $other->getValues;
    for my $n (@{$self->getValues})
    {
	$cmp->pushContext ($n->getNodeName);
	return 0 unless $n->equals ($ov->[$i], $cmp);
	$i++;
	$cmp->popContext;
    }
    return 0 unless $cmp->sameReadOnly ($self->isReadOnly,
					$other->isReadOnly);
    1;
}

package XML::DOM::NodeList;

sub equals
{
    my ($self, $other, $cmp) = @_;
    return 0 unless $cmp->sameType ($self, $other);

    return $cmp->fail("wrong length") 
	unless $self->getLength == $other->getLength;

    my $i = 0;
    for my $n (@$self)
    {
	$cmp->pushContext ("[$i]");
	return 0 unless $n->equals ($other->[$i], $cmp);
	$i++;
	$cmp->popContext;
    }
    1;
}

package XML::DOM::Node;

sub get_prop_byname
{
    my ($self, $propname) = @_;
    my $pkg = ref ($self);

    no strict 'refs';
    my $hfields = \ %{"$pkg\::HFIELDS"};
    $self->[$hfields->{$propname}];
}

sub equals
{
    my ($self, $other, $cmp) = @_;
    return 0 unless $cmp->sameType ($self, $other);

    my $hasKids = $self->hasChildNodes;
    return $cmp->fail("hasChildNodes") unless $hasKids == $other->hasChildNodes;

    if ($hasKids)
    {
	$cmp->pushContext ("C");
	return 0 unless $self->[_C]->equals ($other->[_C], $cmp);
	$cmp->popContext;
    }
    return 0 unless $cmp->sameReadOnly ($self->isReadOnly,
					$other->isReadOnly);

    for my $prop (@{$self->getCmpProps})
    {
	$cmp->pushContext ($prop);	
	my $p1 = $self->get_prop_byname ($prop);
	my $p2 = $other->get_prop_byname ($prop);
	if (ref ($p1))
	{
	    return 0 unless $p1->equals ($p2, $cmp);
	}
	elsif (! defined ($p1))
	{
	    return 0 if defined $p2;
	}
	else
	{
	    return $cmp->fail("$p1 !=  $p2") unless $p1 eq $p2;
	}
	$cmp->popContext;
    }
    1;
}

sub getCmpProps
{
    return [];
}

package XML::DOM::Attr;

sub getCmpProps
{
    ['Name', 'Specified'];
}

package XML::DOM::ProcessingInstruction;

sub getCmpProps
{
    ['Target', 'Data'];
}

package XML::DOM::Notation;

sub getCmpProps
{
    return ['Name', 'Base', 'SysId', 'PubId'];
}

package XML::DOM::Entity;

sub getCmpProps
{
    return ['NotationName', 'Parameter', 'Value', 'SysId', 'PubId'];
}

package XML::DOM::EntityReference;

sub getCmpProps
{
    return ['EntityName', 'Parameter'];
}

package XML::DOM::AttDef;

sub getCmpProps
{
    return ['Name', 'Type', 'Required', 'Implied', 'Quote', 'Default', 'Fixed'];
}

package XML::DOM::AttlistDecl;

sub getCmpProps
{
    return ['ElementName', 'A'];
}

package XML::DOM::ElementDecl;

sub getCmpProps
{
    return ['Name', 'Model'];
}

package XML::DOM::Element;

sub getCmpProps
{
    return ['TagName', 'A'];
}

package XML::DOM::CharacterData;

sub getCmpProps
{
    return ['Data'];
}

package XML::DOM::XMLDecl;

sub getCmpProps
{
    return ['Version', 'Encoding', 'Standalone'];
}

package XML::DOM::DocumentType;

sub getCmpProps
{
    return ['Entities', 'Notations', 'Name', 'SysId', 'PubId', 'Internal'];
}

package XML::DOM::Document;

sub getCmpProps
{
    return ['XmlDecl', 'Doctype'];
}

1;
